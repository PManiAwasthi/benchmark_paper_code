import os
import pickle
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# to load the pickle file
def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
# stats metric
def stats(arr):
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "nan_count": np.isnan(arr).sum(),
        "inf_count": np.isinf(arr).sum(),
    }


# to make the report
def inspect_split(split_name, split):

    report = {}
    
    report["samples"] = len(split["id"])

    report["text_shape"] = split["text"].shape
    report["audio_shape"] = split["audio"].shape
    report["vision_shape"] = split["vision"].shape

    report["text_stats"] = stats(split["text"])
    report["audio_stats"] = stats(split["audio"])
    report["vision_stats"] = stats(split["vision"])

    labels = split["regression_labels"]

    report["label_mean"] = float(np.mean(labels))
    report["label_std"] = float(np.std(labels))
    report["label_min"] = float(np.min(labels))
    report["label_max"] = float(np.max(labels))

    if "audio_lengths" in split:
        report["audio_length_stats"] = {
            "mean": float(np.mean(split["audio_lengths"])),
            "std": float(np.std(split["audio_lengths"])),
            "min": int(np.min(split["audio_lengths"])),
            "max": int(np.max(split["audio_lengths"])),
        }

    if "vision_lengths" in split:
        report["vision_length_stats"] = {
            "mean": float(np.mean(split["vision_lengths"])),
            "std": float(np.std(split["vision_lengths"])),
            "min": int(np.min(split["vision_lengths"])),
            "max": int(np.max(split["vision_lengths"])),
        }
    
    return report

# to save the label histogram
def save_label_hist(labels, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=30, color='blue', alpha=0.7)
    plt.title("Label Distribution")
    plt.xlabel("Label Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, "label_histogram.png"))
    plt.close()


# to export sample examples
def export_sample_examples(split, output_dir, num_samples=5):
    sample_idxs = random.sample(range(len(split["id"])), min(num_samples, len(split["id"])))
    rows = []
    for i in sample_idxs:
        rows.append({
            "id": split["id"][i],
            "text": split["raw_text"][i],
            "label": split["regression_labels"][i],
            "text_shape": split["text"][i].shape,
            "audio_shape": split["audio"][i].shape,
            "vision_shape": split["vision"][i].shape,
            
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "sample_examples.csv"), index=False) 
        

# main function to inspect the dataset
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True
    )

    args = parser.parse_args()

    data = load_pickle_file(args.data)

    dataset_name = os.path.basename(args.data).replace(".pkl", "")

    if "MOSI" in args.data:
        dataset_family = "MOSI"
    elif "MOSEI" in args.data:
        dataset_family = "MOSEI"
    output_dir = os.path.join("inspections", "reports", dataset_family,dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    report = {}

    for split_name in ["train", "valid", "test"]:
        split = data[split_name]
        report[split_name] = inspect_split(split_name, split)
    
    labels = data["train"]["regression_labels"]

    save_label_hist(labels, output_dir)

    export_sample_examples(
        data["train"],
        output_dir
    )
    
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(f"# Dataset Report\n\n")

        for split in report:
            r = report[split]

            f.write(f"## {split.capitalize()} \n\n")

            f.write(f"Samples: {r['samples']} \n\n")

            f.write(f"Text Shape: {r['text_shape']} \n\n")

            f.write(f"Audio Shape: {r['audio_shape']} \n\n")

            f.write(f"Vision Shape: {r['vision_shape']} \n\n")

            f.write(f"Lable Mean: {r['label_mean']:.4f} \n\n")

            f.write(f"Lable Std: {r['label_std']:.4f} \n\n")

    print(f"Report saved to {output_dir}")

# calling main
if __name__ == "__main__":
    main()
