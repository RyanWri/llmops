import torch


def check_torch_gpu():
    print("PyTorch:")
    if torch.cuda.is_available():
        print(f"  GPU(s) Available: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch is using GPU: {torch.cuda.is_initialized()}")
    else:
        print("  No GPU found for PyTorch.")
