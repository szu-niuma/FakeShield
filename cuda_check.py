import torch

# 检查CUDA是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.get_device_name(0)}")

# 验证CUDA计算
if torch.cuda.is_available():
    x = torch.rand(3, 3).cuda()
    print(f"CUDA 张量: {x}")