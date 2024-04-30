import torch
import netron
# 定义 PyTorch 模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc(x)
        return x
# 创建模型实例并加载预训练权重
model = MyModel()
# 设置示例输入
input = torch.randn(1, 3, 32, 32)
# 将模型导出为 ONNX 格式
torch.onnx.export(model, input, './model/onnx_model.onnx')  # 导出后 netron.start(path) 打开
netron.start('./model/onnx_model.onnx')