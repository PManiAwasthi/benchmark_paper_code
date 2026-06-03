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

    output_dir = os.path.join("inspections", "reports", dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    report = {}

    for split_name in ["train", "valid", "test"]:
        split = data[split_name]
        report[split_name] = inspect_split(split_name, split)
        
# calling main
if __name__ == "__main__":
    main()
