import torch

class MetricsExporter:
    def __init__(self):
        self.metrics = {"loss": [], "rewards": []}
        self.gpu_stats = {"memory_allocated": [], "memory_reserved": []}
    
    def log_metric(self, key, value):
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]
    
    def log_gpu_stats(self):
        if torch.cuda.is_available():
            self.gpu_stats["memory_allocated"].append(torch.cuda.memory_allocated())
            self.gpu_stats["memory_reserved"].append(torch.cuda.memory_reserved())
    
    def export(self):
        return {"metrics": self.metrics, "gpu_stats": self.gpu_stats}
