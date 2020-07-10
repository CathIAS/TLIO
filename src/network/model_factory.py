from network.model_resnet import BasicBlock1D, ResNet1D
from network.model_tcn import TlioTcn


def get_model(arch, net_config, input_dim=6, output_dim=3):
    if arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "tcn":
        network = TlioTcn(
            input_dim,
            output_dim,
            [64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU",
        )
    else:
        raise ValueError("Invalid architecture: ", arch)
    return network
