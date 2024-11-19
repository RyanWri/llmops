import subprocess
import tensorflow as tf


def get_device():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPUs detected")
    else:
        print("GPUs detected:")
        for gpu in gpus:
            print(gpu)


def get_gpu_stats():
    try:
        # Run the nvidia-smi command
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check for errors
        if result.returncode != 0:
            print("Error executing nvidia-smi:", result.stderr)
            return

        # Parse the output
        gpus = []
        for line in result.stdout.strip().split("\n"):
            fields = line.split(", ")
            gpu = {
                "index": fields[0],
                "name": fields[1],
                "memory_total": fields[2] + " MB",
                "memory_used": fields[3] + " MB",
                "memory_free": fields[4] + " MB",
                "utilization": fields[5] + " %",
                "temperature": fields[6] + " °C",
            }
            gpus.append(gpu)

        # Display GPU stats
        for gpu in gpus:
            print(f"GPU {gpu['index']} - {gpu['name']}")
            print(f"  Total Memory: {gpu['memory_total']}")
            print(f"  Used Memory: {gpu['memory_used']}")
            print(f"  Free Memory: {gpu['memory_free']}")
            print(f"  GPU Utilization: {gpu['utilization']}")
            print(f"  Temperature: {gpu['temperature']}")
            print()

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    print(tf.test.is_built_with_cuda())
