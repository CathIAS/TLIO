"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch.nn as nn
from network.model_resnet import Bottleneck, conv1x1

class ResNetSeq1D(nn.Module):
    """
    ResNet 1D with dense sequential output. 
    This model is fully-convolutional but is not robust to different input sizes by default.
    in_dim: [in_dim=6, N]
    out_dim: 2x [3, N]
    """

    def __init__(
        self,
        block_type,
        in_dim,
        out_dim,
        group_sizes,
        inter_dim,
        zero_init_residual=False,
    ):
        super(ResNetSeq1D, self).__init__()
        self.base_plane = 64
        self.inplanes = self.base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.residual_groups = nn.Sequential(
            self._make_residual_group1d(block_type, 64, group_sizes[0], stride=1),
            self._make_residual_group1d(block_type, 128, group_sizes[1], stride=2),
            self._make_residual_group1d(block_type, 256, group_sizes[2], stride=2),
            self._make_residual_group1d(block_type, 512, group_sizes[3], stride=2),
        )

        # Output decoder module
        def decoder_():
            return nn.Sequential(
               nn.Sequential(
                   nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm1d(256),
                   nn.ReLU(inplace=True),
                   block_type(256, 256),
               ),
               nn.Sequential(
                   nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm1d(128),
                   nn.ReLU(inplace=True),
                   block_type(128, 128),
               ),
               nn.Sequential(
                   nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm1d(64),
                   nn.ReLU(inplace=True),
                   block_type(64, 64),
               ),
               nn.Sequential(
                   nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=0),
                   nn.BatchNorm1d(32),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=0),
                   nn.BatchNorm1d(16),
                   nn.ReLU(inplace=True),
                   block_type(16, 16),
                   nn.Conv1d(16, out_dim, kernel_size=1, bias=False),
                   nn.BatchNorm1d(out_dim),
               ),
            )

        self.mean_decoder = decoder_()   
        self.logstd_decoder = decoder_()   


        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def run_decoder(self, decoder, x4, x3, x2, x1, x0):
        # Some of the shapes are 1-off because and cant be fixed with padding, 
        # so just add the skips up to the shape and check that it's no more than 1-off
        d3 = decoder[0](x4)
        assert abs(d3.shape[2] - x3.shape[2]) < 2
        d3 = d3 + x3[:,:,:d3.shape[2]]

        d2 = decoder[1](d3)
        assert abs(d2.shape[2] - x2.shape[2]) < 2
        d2 = d2 + x2[:,:,:d2.shape[2]]

        d1 = decoder[2](d2)
        assert abs(d1.shape[2] - x1.shape[2]) < 2
        d1 = d1 + x1[:,:,:d1.shape[2]]

        # No skip for the last one since it was downsampled by 4 at the start 
        # by 7x7 conv and  max pool.
        # It should return 1 less element than the input, since the 0th element would
        # always be 0 displacement we left the padding to be 1-off
        d0 = decoder[3](d1)
        
        return d0

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.input_block(x)

        # Need to store the skip connections
        x1 = self.residual_groups[0](x0)
        x2 = self.residual_groups[1](x1)
        x3 = self.residual_groups[2](x2)
        x4 = self.residual_groups[3](x3)

        # Run the two separate decoders for mean and cov
        mean = self.run_decoder(self.mean_decoder, x4, x3, x2, x1, x0)
        logstd = self.run_decoder(self.logstd_decoder, x4, x3, x2, x1, x0)

        return mean, logstd
