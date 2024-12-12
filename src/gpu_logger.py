import subprocess
import time
import csv


def log_gpu_metrics(interval, log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "GPU Utilization (%)", "Memory Used (MiB)"])

    def log_metrics():
        while True:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            metrics = result.stdout.strip().split(", ")
            timestamp = time.time()
            gpu_utilization = metrics[0]
            memory_used = metrics[1]

            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, gpu_utilization, memory_used])
            time.sleep(interval)

    return log_metrics
