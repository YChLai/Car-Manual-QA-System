import torch

# 在脚本最开始进行检查
print(">>> 正在检查 GPU 环境...")
is_cuda_available = torch.cuda.is_available()
print(f">>> PyTorch 能否找到 CUDA GPU: {is_cuda_available}")
if is_cuda_available:
    print(f">>> GPU 设备名称: {torch.cuda.get_device_name(0)}")
else:
    print(">>> 未找到 CUDA GPU，嵌入模型将使用 CPU 运行。")
print("-" * 20)
