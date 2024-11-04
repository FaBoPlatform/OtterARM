import torch
import torchvision

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print("PyTorchは、CUDAデバイスの認識に成功しました!")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print("PyTorchは、MPSデバイスの認識に成功しました!")
else:
    device = torch.device("cpu")
    print("GPUの認識に失敗しました。PyTorchの再インストールをおこなってください。")

# テンソルの作成と計算
x = torch.randn(5, 3).to(device)
y = torch.randn(5, 3).to(device)
z = x + y
print(z)
