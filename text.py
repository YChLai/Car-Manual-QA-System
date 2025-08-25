import torch

if torch.cuda.is_available():
    print(f"成功！PyTorch可以找到你的GPU。")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("失败。PyTorch无法找到GPU，仍在CPU模式下运行。")