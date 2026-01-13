import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image


def main():
    data_dir = "data"
    batch_size = 64
    latent_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(time.time()))


    fc_layer = nn.Linear(64, 8192).to(device)  # in_features=64, out_features=8192

    convT_layers = [
        nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(64, 3, 5, 2, 2, output_padding=1).to(device)
    ]

    bn_layers = [
        nn.BatchNorm2d(256).to(device),
        nn.BatchNorm2d(128).to(device),
        nn.BatchNorm2d(64).to(device)
    ]

    tanh = nn.Tanh().to(device)


    fc_weight = np.loadtxt(os.path.join(data_dir, "decoder_fc.weight.csv"), delimiter=",")

    fc_layer.weight.data = torch.tensor(
        fc_weight,
        dtype=torch.float32
    ).to(device)

    fc_layer.bias.data = torch.tensor(
        np.loadtxt(os.path.join(data_dir, "decoder_fc.bias.csv"), delimiter=","),
        dtype=torch.float32
    ).to(device)

    convT_indices = [0, 3, 6, 9]
    for i, idx in enumerate(convT_indices):
        weight = np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.weight.csv"), delimiter=",")
        convT_layers[i].weight.data = torch.tensor(
            weight.reshape(convT_layers[i].in_channels, convT_layers[i].out_channels, 5, 5),
            dtype=torch.float32
        ).to(device)

        bias = np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.bias.csv"), delimiter=",")
        convT_layers[i].bias.data = torch.tensor(bias, dtype=torch.float32).to(device)

    bn_indices = [1, 4, 7]
    for i, idx in enumerate(bn_indices):
        bn_layers[i].weight.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.weight.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].bias.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.bias.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].running_mean.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.running_mean.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].running_var.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.running_var.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)


    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        x = fc_layer(z)
        x = x.view(-1, 512, 4, 4)
        print(f"Shape after view: {x.shape}")


        x = convT_layers[0](x)
        print(f"Shape after convT layer 0: {x.shape}")
        x = bn_layers[0](x)
        x = torch.relu(x)

        x = convT_layers[1](x)
        print(f"Shape after convT layer 1: {x.shape}")
        x = bn_layers[1](x)
        x = torch.relu(x)

        x = convT_layers[2](x)
        print(f"Shape after convT layer 2: {x.shape}")
        x = bn_layers[2](x)
        x = torch.relu(x)

        x = convT_layers[3](x)
        print(f"Shape after convT layer 3: {x.shape}")

        output = tanh(x)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        save_image(output, "generated.png", nrow=8)


if __name__ == "__main__":
    main()
