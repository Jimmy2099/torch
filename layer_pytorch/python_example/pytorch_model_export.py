import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 2:

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(scores)
        out = torch.bmm(attention, values)
        return out

class ComplexMLP(nn.Module):
    def __init__(self, input_size=10, num_classes=5):
        super(ComplexMLP, self).__init__()

        self.initial_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.res_block1 = self._make_residual_block(128, 256)
        self.res_block2 = self._make_residual_block(256, 256)
        self.res_block3 = self._make_residual_block(256, 512)

        self.attention = AttentionBlock(512)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def _make_residual_block(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, downsample=downsample)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = x.unsqueeze(-1)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = x.squeeze(-1)

        x = self.attention(x)
        x = x.squeeze(1)

        x = self.fc(x)
        return x

X = torch.randn(100, 10)
y = torch.randn(100, 5)

model = ComplexMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "complex_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)


scripted_model = torch.jit.script(model)
scripted_model.save("complex_model_scripted.pt")

print("Model exported to complex_model.onnx and complex_model_scripted.pt")
