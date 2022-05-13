import torch
import torch.nn as nn


class FSRCNN_model(nn.Module):
    def __init__(self, scale: int) -> None:
        super(FSRCNN_model, self).__init__()

        if scale not in [2, 3, 4]:
            raise ValueError("must be 2, 3 or 4")

        d = 56
        s = 12

        self.feature_extract = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.feature_extract.weight)
        nn.init.zeros_(self.feature_extract.bias)

        self.activation_1 = nn.PReLU(num_parameters=d)

        self.shrink = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1)
        nn.init.kaiming_normal_(self.shrink.weight)
        nn.init.zeros_(self.shrink.bias)

        self.activation_2 = nn.PReLU(num_parameters=s)
        
        # m = 4
        self.map_1 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_1.weight)
        nn.init.zeros_(self.map_1.bias)

        self.map_2 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_2.weight)
        nn.init.zeros_(self.map_2.bias)

        self.map_3 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_3.weight)
        nn.init.zeros_(self.map_3.bias)

        self.map_4 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_4.weight)
        nn.init.zeros_(self.map_4.bias)

        self.activation_3 = nn.PReLU(num_parameters=s)

        self.expand = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1)
        nn.init.kaiming_normal_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

        self.activation_4 = nn.PReLU(num_parameters=d)

        self.deconv = nn.ConvTranspose2d(in_channels=d, out_channels=3, kernel_size=9, 
                                        stride=scale, padding=4, output_padding=scale-1)
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, X_in):
        X = self.feature_extract(X_in)
        X = self.activation_1(X)

        X = self.shrink(X)
        X = self.activation_2(X)

        X = self.map_1(X)
        X = self.map_2(X)
        X = self.map_3(X)
        X = self.map_4(X)
        X = self.activation_3(X)

        X = self.expand(X)
        X = self.activation_4(X)

        X = self.deconv(X)
        X_out = torch.clip(X, 0.0, 1.0)

        return X_out
