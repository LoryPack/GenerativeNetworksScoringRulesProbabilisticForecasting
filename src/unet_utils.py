import torch.nn as nn
import torch.nn.functional as F
from torchtyping import patch_typeguard

patch_typeguard()  # use before @typechecked


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)


class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()
        self.dropout = dropout
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        # if dropout:
        #     assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
        #     layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        if self.dropout:
            # not deactivated by .eval()
            x = F.dropout2d(x, p=self.dropout)
        return x


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()
        self.dropout = dropout
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        # in this way the dropout layer is deactivated by nn.eval()
        # if dropout:
        #     assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
        #     layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        if self.dropout:
            # not deactivated by .eval()
            x = F.dropout2d(x, p=self.dropout)
        return x


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()
        self.dropout = dropout
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        # if dropout:
        #     assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
        #     layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        x = self.center(x)
        if self.dropout:
            # not deactivated by .eval()
            x = F.dropout2d(x, p=self.dropout)
        return x


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()
        self.dropout = dropout
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        # if dropout:
        #     assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
        #     layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """ z is noise"""
        x = self.decoder(x)
        if self.dropout:
            # not deactivated by .eval()
            x = F.dropout2d(x, p=self.dropout)
        return x


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1)
        ]
        if softmax:
            layers.append(nn.Softmax(dim=1))

        self.last = nn.Sequential(*layers)

    def forward(self, x):
        return self.last(x)
