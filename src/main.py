from src.metrics.gpu import get_gpu_stats
from src.metrics.validation import check_torch_gpu


if __name__ == "__main__":
    print("-" * 30)
    check_torch_gpu()
    get_gpu_stats()
