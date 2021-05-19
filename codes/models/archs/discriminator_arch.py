from torch import nn as nn

def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class MultiDiscriminator_d(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator_d, self).__init__()

        # def discriminator_block(in_filters, out_filters, normalize=True):
        #     """Returns downsampling layers of each discriminator block"""
        #     layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        #     if normalize:
        #         layers.append(nn.InstanceNorm2d(out_filters))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        def discriminator_block(in_filters, out_filters, normalization=False):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            layers.append(nn.LeakyReLU(0.2))
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
                    nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.InstanceNorm2d(16, affine=True),
                    *discriminator_block(16, 32),
                    *discriminator_block(32, 64),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 128),
                    nn.Conv2d(128, 1, 8, padding=0)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
