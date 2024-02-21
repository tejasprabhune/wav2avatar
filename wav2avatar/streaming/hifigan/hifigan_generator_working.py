import logging

import torch
import torch.nn.functional as F

import wav2avatar.streaming.hifigan.layers as layers

class HiFiGANGenerator(torch.nn.Module):

    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        kernel_size=3,
        upsample_scales=(2,),
        upsample_kernel_sizes=(2,),
        paddings=None,
        output_paddings=None,
        resblock_kernel_sizes=(3, 7, 11, 15),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        ar_input=600, 
        ar_hidden=256, 
        ar_output=128,
        use_tanh=False,
        use_mlp_ar=True
    ):
        super().__init__()

        paddings = [upsample_scales[i] // 2 + upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]
        output_paddings = [upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]

        if use_mlp_ar:
            self.ar_model = layers.PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)
        else:
            self.ar_conv = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    12,
                    128,
                    3,
                    1,
                    padding=1
                )
            )
            self.ar_linear = torch.nn.Linear(6400, 128)

        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels + ar_output,
            channels // 2,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=paddings[i],
                        output_padding=output_paddings[i],
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    layers.HiFiGANResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        
        if use_tanh:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
            )
        

        if use_weight_norm:
            self.apply_weight_norm()

        self.reset_parameters()
    
    def forward(self, x, ar):

        # x: B, C, seqlen
        # ar: (B, 12, 50)

        #print("x shape:", x.shape) # B, C, seqlen

        ar_feats = self.ar_model(ar) # B, 128
        ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2]) # B, 128, seqlen

        x = torch.cat((x, ar_feats), dim=1) # B, 640, seqlen
        # print(x.shape)

        x = self.input_conv(x) # B, 512, seqlen
        #print("inputconv:", x.shape)

        for i in range(self.num_upsamples):
            #x = self.upsamples[i](x)
            #print(x.shape)

            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](x)
            # print('cs', cs.shape)
            x = cs / self.num_blocks  # (batch_size, some_channels, length)
        
        out = self.output_conv(x)

        return out
    
    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
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



if __name__ == "__main__":
    gan = HiFiGANGenerator()

    print(gan)