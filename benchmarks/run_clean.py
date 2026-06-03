from MMSA import MMSA_run
import os
import json
import time
import argparse

import pandas as pd


SEEDS = [
    1111,
    1112,
    1113,
    1114,
    1115
]

MODELS = [
    "TFN",
    "MulT",
    "MISA",
    "SELF_MM",
    "CENET",
    "ALMT"
]


def run_experiment(
    dataset,
    model,
    seed,
    gpu_id=0
):

    print(
        f"Running {model} | {dataset} | seed={seed}"
    )

    start_time = time.time()

    MMSA_run(
        model.lower(),
        dataset.lower(),
        seeds=[seed],
        gpu_ids=[gpu_id]
    )

    training_time = time.time() - start_time

    metrics = {
        "seed": seed,
        "Training_Time": training_time
    }

    return metrics


def aggregate(results):

    df = pd.DataFrame(results)

    summary = {}

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    for col in numeric_cols:

        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()

    return summary


def run_model(
    dataset,
    model,
    gpu_id=0
):

    print(
        f"\nRunning {model} on {dataset}\n"
    )

    all_results = []

    for seed in SEEDS:

        metrics = run_experiment(
            dataset,
            model,
            seed,
            gpu_id
        )

        all_results.append(metrics)

    output_dir = os.path.join(
        "results",
        "clean_summary",
        dataset
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    pd.DataFrame(
        all_results
    ).to_csv(
        os.path.join(
            output_dir,
            f"{model}_all_seeds.csv"
        ),
        index=False
    )

    summary = aggregate(
        all_results
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            output_dir,
            f"{model}.csv"
        ),
        index=False
    )

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=str
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None
    )

    parser.add_argument(
        "--all_models",
        action="store_true"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.all_models:

        models = MODELS

    elif args.model is not None:

        models = [args.model]

    else:

        raise ValueError(
            "Specify either --model or --all_models"
        )

    benchmark_rows = []

    for model in models:

        summary = run_model(
            args.dataset,
            model,
            args.gpu
        )

        summary["model"] = model

        benchmark_rows.append(
            summary
        )

    output_dir = os.path.join(
        "results",
        "clean_summary",
        args.dataset
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    pd.DataFrame(
        benchmark_rows
    ).to_csv(
        os.path.join(
            output_dir,
            "benchmark_matrix.csv"
        ),
        index=False
    )

    print(
        "\nClean benchmark complete!"
    )


if __name__ == "__main__":
    main()