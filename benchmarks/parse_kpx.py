import re
import numpy as np
import sys

def parse_kpax_logs(file_path):
    # Lists to store our parsed data
    times = []
    iters = []
    sizes = []

    # Regex to capture: time (float), iterations (int), and tree size (int)
    pattern = re.compile(
        r"execution time: ([\d.]+) seconds\. Iterations: (\d+)\. Tree Size: (\d+)"
    )

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
                    iters.append(int(match.group(2)))
                    sizes.append(int(match.group(3)))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    if not iters:
        print("No valid log lines found. Check your file format.")
        return

    # Convert to numpy arrays for easy math
    data = {
        "Runtime (s)": np.array(times),
        "Iterations": np.array(iters),
        "Tree Size": np.array(sizes)
    }

    # Print Header
    print(f"{'Metric':<15} | {'Min':<10} | {'Avg':<10} | {'Median':<10} | {'Max':<10} | {'Std':<10}")
    print("-" * 75)

    for label, values in data.items():
        v_min = np.min(values)
        v_avg = np.mean(values)
        v_med = np.median(values)
        v_max = np.max(values)
        v_std = np.std(values)

        # Formatting floats for runtime and ints for tree stats
        if "Runtime" in label:
            print(f"{label:<15} | {v_min:<10.4f} | {v_avg:<10.4f} | {v_med:<10.4f} | {v_max:<10.4f} | {v_std:<10.4f}")
        else:
            print(f"{label:<15} | {v_min:<10.0f} | {v_avg:<10.2f} | {v_med:<10.1f} | {v_max:<10.0f} | {v_std:<10.2f}")

if __name__ == "__main__":
    # Change 'logs.txt' to your actual filename
    filename = "benchmarks/logs.txt" 
    parse_kpax_logs(filename)