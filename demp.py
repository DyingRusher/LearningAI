import torch

print(torch.cuda.get_device_properties(0).total_memory/1024 ** 3)