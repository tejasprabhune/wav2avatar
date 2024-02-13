import torch

class PastFCEncoder(torch.nn.Module):
    '''
    Autoregressive class in CARGAN
    https://github.com/descriptinc/cargan/blob/master/cargan/model/condition.py#L6
    '''
    def __init__(self, input_len=512, hidden_dim=256, output_dim=128):
        '''
        Args:
            input_len: the number of samples of autoregressive conditioning
        '''
        super().__init__()

        model = [
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.LeakyReLU(.1)]
        for _ in range(3):
            model.extend([
                torch.nn.Linear(
                    hidden_dim,
                    hidden_dim),
                torch.nn.LeakyReLU(.1)])
        model.append(
            torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*model)
    
    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, 1, input_len)

        Return:
            shape (batch_size, output_dim)
        '''
        x = x.reshape(x.shape[0], -1)
        return self.model(x)

class HiFiGANResidualBlock(torch.nn.Module):
    """
    Residual block module in HiFiGAN. 
    From https://github.com/articulatory/articulatory.
    """

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize HiFiGANResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            self.convs1 += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x