import torch

# 檢查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    # 顯示GPU的名稱
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 顯示GPU的記憶體容量和使用情況
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Cached GPU Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

# 確認PyTorch是否在GPU上運行
x = torch.randn(1, 1).to(device)
print(f"PyTorch is running on {x.device}")
