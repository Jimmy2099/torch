import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import time

def main():
    # 基础配置
    data_dir = "data"
    batch_size = 64
    latent_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(time.time()))

    # ====================== 参数加载 ======================
    # 1. 全连接层参数
    fc_weight = np.loadtxt(os.path.join(data_dir, "decoder_fc.weight.csv"), delimiter=",")
    fc_weight = torch.tensor(fc_weight.T, dtype=torch.float32).to(device)  # [64, 8192]
    fc_bias = np.loadtxt(os.path.join(data_dir, "decoder_fc.bias.csv"), delimiter=",")
    fc_bias = torch.tensor(fc_bias, dtype=torch.float32).to(device)  # [8192]

    # 2. 转置卷积层参数（严格按照层顺序）
    convT_params = [
        {  # decoder_conv.0
            "weight": np.loadtxt(os.path.join(data_dir, "decoder_conv.0.weight.csv"), delimiter=","),
            "bias": np.loadtxt(os.path.join(data_dir, "decoder_conv.0.bias.csv"), delimiter=","),
            "in_ch": 512,
            "out_ch": 256
        },
        {  # decoder_conv.3
            "weight": np.loadtxt(os.path.join(data_dir, "decoder_conv.3.weight.csv"), delimiter=","),
            "bias": np.loadtxt(os.path.join(data_dir, "decoder_conv.3.bias.csv"), delimiter=","),
            "in_ch": 256,
            "out_ch": 128
        },
        {  # decoder_conv.6
            "weight": np.loadtxt(os.path.join(data_dir, "decoder_conv.6.weight.csv"), delimiter=","),
            "bias": np.loadtxt(os.path.join(data_dir, "decoder_conv.6.bias.csv"), delimiter=","),
            "in_ch": 128,
            "out_ch": 64
        },
        {  # decoder_conv.9
            "weight": np.loadtxt(os.path.join(data_dir, "decoder_conv.9.weight.csv"), delimiter=","),
            "bias": np.loadtxt(os.path.join(data_dir, "decoder_conv.9.bias.csv"), delimiter=","),
            "in_ch": 64,
            "out_ch": 3
        }
    ]

    # 处理转置卷积权重维度
    for param in convT_params:
        param["weight"] = torch.tensor(
            param["weight"].reshape(param["in_ch"], param["out_ch"], 5, 5),  # 关键维度调整
            dtype=torch.float32
        ).to(device)
        param["bias"] = torch.tensor(param["bias"], dtype=torch.float32).to(device)

    # 3. BatchNorm参数（严格对应层顺序）
    bn_params = [
        {  # decoder_conv.1
            "gamma": np.loadtxt(os.path.join(data_dir, "decoder_conv.1.weight.csv"), delimiter=","),
            "beta": np.loadtxt(os.path.join(data_dir, "decoder_conv.1.bias.csv"), delimiter=","),
            "mean": np.loadtxt(os.path.join(data_dir, "decoder_conv.1.running_mean.csv"), delimiter=","),
            "var": np.loadtxt(os.path.join(data_dir, "decoder_conv.1.running_var.csv"), delimiter=","),
            "channels": 256
        },
        {  # decoder_conv.4
            "gamma": np.loadtxt(os.path.join(data_dir, "decoder_conv.4.weight.csv"), delimiter=","),
            "beta": np.loadtxt(os.path.join(data_dir, "decoder_conv.4.bias.csv"), delimiter=","),
            "mean": np.loadtxt(os.path.join(data_dir, "decoder_conv.4.running_mean.csv"), delimiter=","),
            "var": np.loadtxt(os.path.join(data_dir, "decoder_conv.4.running_var.csv"), delimiter=","),
            "channels": 128
        },
        {  # decoder_conv.7
            "gamma": np.loadtxt(os.path.join(data_dir, "decoder_conv.7.weight.csv"), delimiter=","),
            "beta": np.loadtxt(os.path.join(data_dir, "decoder_conv.7.bias.csv"), delimiter=","),
            "mean": np.loadtxt(os.path.join(data_dir, "decoder_conv.7.running_mean.csv"), delimiter=","),
            "var": np.loadtxt(os.path.join(data_dir, "decoder_conv.7.running_var.csv"), delimiter=","),
            "channels": 64
        }
    ]

    # 转换BN参数为Tensor
    for param in bn_params:
        for key in ["gamma", "beta", "mean", "var"]:
            param[key] = torch.tensor(param[key], dtype=torch.float32).to(device)

    # ====================== 手动前向传播 ======================
    def decode(z):
        # 全连接层
        x = F.linear(z, fc_weight.t(), fc_bias)  # 使用转置权重
        x = x.view(-1, 512, 4, 4)
        print("After fc:", x.shape)  # [64,512,4,4]

        # 第一组转置卷积
        x = F.conv_transpose2d(
            x,
            weight=convT_params[0]["weight"],
            bias=convT_params[0]["bias"],
            stride=2,
            padding=2,
            output_padding=1
        )
        print("After convT0:", x.shape)  # [64,256,8,8]

        # 第一个BN+ReLU
        x = F.batch_norm(
            x,
            bn_params[0]["mean"],
            bn_params[0]["var"],
            weight=bn_params[0]["gamma"],
            bias=bn_params[0]["beta"],
            training=False
        )
        x = F.relu(x)
        print("After BN1:", x.shape)  # [64,256,8,8]

        # 第二组转置卷积
        x = F.conv_transpose2d(
            x,
            weight=convT_params[1]["weight"],
            bias=convT_params[1]["bias"],
            stride=2,
            padding=2,
            output_padding=1
        )
        print("After convT3:", x.shape)  # [64,128,16,16]

        # 第二个BN+ReLU
        x = F.batch_norm(
            x,
            bn_params[1]["mean"],
            bn_params[1]["var"],
            weight=bn_params[1]["gamma"],
            bias=bn_params[1]["beta"],
            training=False
        )
        x = F.relu(x)
        print("After BN4:", x.shape)  # [64,128,16,16]

        # 第三组转置卷积
        x = F.conv_transpose2d(
            x,
            weight=convT_params[2]["weight"],
            bias=convT_params[2]["bias"],
            stride=2,
            padding=2,
            output_padding=1
        )
        print("After convT6:", x.shape)  # [64,64,32,32]

        # 第三个BN+ReLU
        x = F.batch_norm(
            x,
            bn_params[2]["mean"],
            bn_params[2]["var"],
            weight=bn_params[2]["gamma"],
            bias=bn_params[2]["beta"],
            training=False
        )
        x = F.relu(x)
        print("After BN7:", x.shape)  # [64,64,32,32]

        # 最后一层转置卷积
        x = F.conv_transpose2d(
            x,
            weight=convT_params[3]["weight"],
            bias=convT_params[3]["bias"],
            stride=2,
            padding=2,
            output_padding=1
        )
        print("After convT9:", x.shape)  # [64,3,64,64]

        return torch.tanh(x)

    # ====================== 生成图像 ======================
    z = torch.randn(batch_size, latent_dim).to(device)
    with torch.no_grad():
        output = decode(z)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        save_image(output, "generated.png", nrow=8)

if __name__ == "__main__":
    main()
