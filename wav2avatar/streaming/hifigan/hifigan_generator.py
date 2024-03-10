import logging

import torch
import torch.nn.functional as F

import wav2avatar.streaming.hifigan.layers as layers

class HiFiGANGenerator(torch.nn.Module):

    def __init__(
        self,
        in_channels=512,
        res_channels=256,
        kernel_size=3,
        context_len=50,
        out_len=5,
        ema_channels=12,
        ar_len=50,
        ar_ft_output=128,
        ar_hidden=256,
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1}
    ):
        super().__init__()

        # AR MODEL

        ar_input = ar_len * ema_channels

        self.ar_model = layers.PastFCEncoder(
            input_len=ar_input,
            hidden_dim=ar_hidden,
            output_dim=ar_ft_output
        )

        # AUDIO + AR MODEL

        self.input_conv = torch.nn.Conv1d(
            in_channels + ar_ft_output,
            res_channels,
            kernel_size,
            1,
            padding=1
        )

        self.blocks = torch.nn.ModuleList()
        for j in range(len(resblock_kernel_sizes)):
            self.blocks += [
                layers.HiFiGANResidualBlock(
                    kernel_size=resblock_kernel_sizes[j],
                    channels=res_channels,
                    dilations=resblock_dilations[j],
                    bias=bias,
                    use_additional_convs=use_additional_convs,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]

        self.output_conv = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                in_channels=res_channels,
                out_channels=ema_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            ),
            # torch.nn.Linear(in_features=context_len, out_features=res_channels),
            # torch.nn.Linear(in_features=res_channels, out_features=out_len)
        )
        #self.ema_conv = torch.nn.Sequential(
        #    torch.nn.LeakyReLU(),
        #    torch.nn.Conv1d(
        #        in_channels=context_len,
        #        out_channels=out_len,
        #        kernel_size=kernel_size,
        #        stride=1,
        #        padding=1
        #    )
        #)
    
        self.apply_weight_norm()
        self.reset_parameters()
    
    def forward(self, audio, ar):

        ar_feats = self.ar_model(ar)
        ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, audio.shape[2])


        x = torch.cat((audio, ar_feats), dim=1)
        x = self.input_conv(x)


        cs = 0.0
        for block in self.blocks:
            cs += block(x)

        x = cs / len(self.blocks)


        ema = self.output_conv(x)


        return ema[:, :, -5:]

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.parametrizations.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)
    
    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")
