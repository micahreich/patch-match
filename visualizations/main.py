import re
import json
from benchmark_data import benchmark_text
import matplotlib.pyplot as plt
import scienceplots

def parse_benchmark_data(text):
    # Split the text into sections based on thread count
    sections = re.split(r'\n(?=\d+ Threads:)', text)
    benchmark = {}

    for section in sections:
        # Find the thread count
        threads_match = re.search(r'(\d+) Threads:', section)
        if not threads_match:
            continue
        thread_count = threads_match.group(1)

        # Find all the run data
        runs = re.findall(r'(\[\s*={60}\s*\] 100 %\n\n.*?)(?=\n\n\n|\Z)', section, re.DOTALL)
        run_data = []

        for run in runs:
            data = {}

            # Extract data for each level
            levels = re.findall(r'Level (\d) times:\n.*?average ANN time:\s+([\d.]+) ms.*?average reconstruction time:\s+([\d.]+) ms.*?total time:\s+([\d.]+) ms', run, re.DOTALL)
            for level, ann_time, recon_time, total_time in levels:
                data[f"Level {level}"] = {
                    "Average ANN Time": float(ann_time),
                    "Average Reconstruction Time": float(recon_time),
                    "Total Time": float(total_time)
                }
            # print(run)
            # Extract other data
            pyramid_time = re.search(r'Pyramid build time:\s+([\d.]+) ms', run).group(1)
            init_time = re.search(r'Initialization time:\s+([\d.]+) ms', run).group(1)
            total_ann_time = re.search(r'Total average ANN time:\s+([\d.]+) ms', run).group(1)
            total_recon_time = re.search(r'Total average reconstruction time:\s+([\d.]+) ms', run).group(1)
            total_run_time = re.search(r'Total time:\s+([\d.]+) s', run).group(1)

            data.update({
                "Pyramid Build Time": float(pyramid_time),
                "Initialization Time": float(init_time),
                "Total Average ANN Time": float(total_ann_time),
                "Total Average Reconstruction Time": float(total_recon_time),
                "Total Time": total_run_time
            })

            run_data.append(data)

        benchmark[f"{thread_count} Threads"] = run_data

    return json.loads(json.dumps(benchmark, indent=4))

# Example usage

parsed_json = parse_benchmark_data(benchmark_text)
# print(parsed_json.values())

def graph_time(level, feature):
    plt.style.use('science')

    thread_counts = [int(key.split()[0]) for key in parsed_json.keys()]
    level_times = [round(run[0][f"Level {level}"][feature] / 1000, 2) for run in parsed_json.values()]

    plt.plot(thread_counts, level_times)
    plt.xlabel(f'Number of Threads')
    plt.ylabel(f'Level {level} {feature} (s)')
    plt.title(f'Level {level} {feature} vs Number of Threads')
    plt.show()

graph_time(0, "Average ANN Time")

