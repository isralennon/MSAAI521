import torch

def test_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"✓ GPU setup complete - {gpu_count} GPU(s) available")
        return gpu_count
    else:
        print("⚠ No GPU available")
        return 0

