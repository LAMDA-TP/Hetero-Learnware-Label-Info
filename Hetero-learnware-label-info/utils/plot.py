import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_dataset_name(dataset_name):
    parts = dataset_name.split("__")
    return parts[1]

def format_func(x, pos):
    x_copy = int(x)
    if x_copy >= 1000:
        return f"{x_copy/1000:.0f}k"
    else:
        return str(x_copy)

def aggregate_res_info(results_dict, res_path):
    data = []
    columns = [
        "Task name",
        "Our method Mean",
        "Our method Std",
        "Our method Summary",
        "Self training Mean",
        "Self training Std",
        "Self training Summary",
        "User ensemble Mean",
        "User ensemble Std",
        "User ensemble Summary",
    ]
    for task_name, res in results_dict.items():
        mean, std = (
            res["Our method"][0],
            res["Our method"][1],
        )
        formatted_mean = "{:.3f}".format(mean)
        formatted_std = "{:.3f}".format(std)
        summary_str = f"{formatted_mean} ({formatted_std})"
        st_mean, st_std = (
            res["Self-training"][0],
            res["Self-training"][1],
        )
        st_formatted_mean = "{:.3f}".format(st_mean)
        st_formatted_std = "{:.3f}".format(st_std)
        st_summary_str = f"{st_formatted_mean} ({st_formatted_std})"
        ue_mean, ue_std = (
            res["Ensemble with user"][0],
            res["Ensemble with user"][1],
        )
        ue_formatted_mean = "{:.3f}".format(ue_mean)
        ue_formatted_std = "{:.3f}".format(ue_std)
        ue_summary_str = f"{ue_formatted_mean} ({ue_formatted_std})"
        data.append(
            [
                task_name,
                formatted_mean,
                formatted_std,
                summary_str,
                st_formatted_mean,
                st_formatted_std,
                st_summary_str,
                ue_mean,
                ue_std,
                ue_summary_str,
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(res_path, index=False)
    return df


def aggregate_results(
    results_dict,
    task_type="classification",
    n_sample_labeled_data_list=[
        10,
        50,
        100,
        500,
        1000,
        2000,
        5000,
    ]
):
    our_method_performacne_list = []
    self_training_performance_list = []
    user_ensemble_performance_list = []
    win_list = []
    win_ratio_list = []
    valid_count_list = []
    for n_sample in n_sample_labeled_data_list:
        current_our_method_performance = []
        current_self_training_performance = []
        current_user_ensemble_performance = []
        win_count = 0
        valid_count = 0
        for results in results_dict.values():
            if n_sample in results:
                valid_count += 1
                current_our_method_performance.append(results[n_sample]["Our method"])
                current_self_training_performance.append(
                    results[n_sample]["Self-training"]
                )
                current_user_ensemble_performance.append(
                    results[n_sample]["Ensemble with user"]
                )
                if task_type == "classification":
                    if (
                        results[n_sample]["Ensemble with user"]
                        > results[n_sample]["Self-training"]
                    ):
                        win_count += 1
                else:
                    if (
                        results[n_sample]["Ensemble with user"]
                        < results[n_sample]["Self-training"]
                    ):
                        win_count += 1

        our_method_performacne_list.append(np.mean(current_our_method_performance))
        self_training_performance_list.append(
            np.mean(current_self_training_performance)
        )
        user_ensemble_performance_list.append(
            np.mean(current_user_ensemble_performance)
        )
        win_list.append(win_count)
        win_ratio_list.append(win_count / valid_count)
        valid_count_list.append(valid_count)

    combined_results = {
        "our_method_performacne_list": our_method_performacne_list,
        "self_training_performance_list": self_training_performance_list,
        "user_ensemble_performance_list": user_ensemble_performance_list,
        "win_list": win_list,
        "win_ratio_list": win_ratio_list,
        "valid_count_list": valid_count_list,
    }
    return combined_results


def plot_user_curve_aggregation_classification(results_dict, fig_dir):
    plt.figure(figsize=(13.5, 6.75))
    n_col, n_row = 4, 3
    select_dataset_name_list = [
        "openml__mfeat-karhunen__16",
        "openml__gina_agnostic__3891",
        "openml__Bioresponse__9910",
        "openml__christine__168908",
        "openml__eye_movements__3897",
        "openml__nursery__9892",
        "openml__magic__146206",
        "openml__nomao__9977",
        "openml__volkert__168331",
    ]
    for i, dataset_name in enumerate(select_dataset_name_list):
        valid_sample_size_list = list(results_dict[dataset_name].keys())
        results_df = pd.DataFrame(results_dict[dataset_name]).T

        if i < 3:
            idx = i + 1
        elif i < 6:
            idx = i + 2
        else:
            idx = i + 3

        plt.subplot(n_row, n_col, idx)
        plt.plot(
            results_df["Self-training"].to_numpy(),
            label="Self-training",
            linestyle="--",
            marker="s",
        )
        plt.plot(
            results_df["Ensemble with user"].to_numpy(),
            label="Ensemble",
            linestyle="-.",
            marker="^",
        )
        plt.xticks(
            range(len(valid_sample_size_list)),
            [format_func(x, None) for x in valid_sample_size_list],
        )
        plt.xlabel("Number of labeled data")
        plt.ylabel("Accuracy")
        plt.title(extract_dataset_name(dataset_name))
        plt.legend()

    dataset_name_list = [
        [
            "openml__credit-g__31",
            "openml__semeion__9964",
            "openml__mfeat-karhunen__16",
            "openml__splice__45",
            "openml__gina_agnostic__3891",
            "openml__Bioresponse__9910",
        ],
        [
            "openml__sylvine__168912",
            "openml__christine__168908",
            "openml__first-order-theorem-proving__9985",
            "openml__satimage__2074",
            "openml__fabert__168910",
            "openml__GesturePhaseSegmentationProcessed__14969",
            "openml__robert__168332",
            "openml__artificial-characters__14964",
            "openml__eye_movements__3897",
            "openml__nursery__9892",
            "openml__eeg-eye-state__14951",
        ],
        [
            "openml__magic__146206",
            "openml__riccardo__168338",
            "openml__guillermo__168337",
            "openml__nomao__9977",
            "openml__Click_prediction_small__190408",
            "openml__volkert__168331",
        ],
    ]
    n_sample_lists = [
        [10, 50, 100, 500, 1000, 2000],
        [10, 50, 100, 500, 1000, 2000, 5000],
        [10, 50, 100, 500, 1000, 2000, 5000],
    ]
    title_list = [
        "Small dataset (<5k)",
        "Medium dataset (5k-1.5k)",
        "Large dataset (1.5k-60k)",
    ]

    for idx in [4, 8, 12]:
        current_sample_list = n_sample_lists[int(idx / 4) - 1]
        combined_results = aggregate_results(
            [
                results_dict[dataset_name]
                for dataset_name in dataset_name_list[int(idx / 4) - 1]
            ],
            "classification",
            current_sample_list,
        )
        win_list = combined_results["win_list"]
        win_ratio_list = combined_results["win_ratio_list"]
        valid_count_list = combined_results["valid_count_list"]

        color = "tab:orange"
        ax1 = plt.subplot(n_row, n_col, idx)
        ax1.set_xlabel("Number of labeled data")
        ax1.set_ylabel("Win ratio", color=color)
        (line1,) = ax1.plot(
            range(len(current_sample_list)),
            win_ratio_list,
            color=color,
            linestyle="-",
            marker="o",
            label="Ensemble",
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(range(len(current_sample_list)))
        ax1.set_xticklabels([format_func(x, None) for x in current_sample_list])

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Dataset count", color=color)
        bar1 = ax2.bar(
            range(len(current_sample_list)),
            valid_count_list,
            color=color,
            alpha=0.5,
            label="Valid",
        )
        bar2 = ax2.bar(
            range(len(current_sample_list)),
            win_list,
            color="tab:blue",
            alpha=0.5,
            label="Win",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        lines = [line1]
        bars = [bar1, bar2]
        labels = [line.get_label() for line in lines] + [
            bar.get_label() for bar in bars
        ]
        plt.legend(lines + bars, labels, loc="lower left", fontsize=8, framealpha=0.5)
        plt.title(title_list[int(idx / 4) - 1])

    fig_path = os.path.join(
        fig_dir, f"user_performance_curve_classification_aggregation.pdf"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)


def plot_user_curve_aggregation_regression(
    results_dict,
    fig_dir,
    n_sample_labeled_data_list=[
        10,
        50,
        100,
        500,
        1000,
        2000,
        5000,
    ]
):
    plt.figure(figsize=(13.5, 4.5))
    n_col, n_row = 4, 2
    for i, (dataset_name, result) in results_dict.items():
        valid_sample_size_list = list(result.keys())
        results_df = pd.DataFrame(result).T

        plt.subplot(n_row, n_col, i + 1)
        plt.plot(
            results_df["Self-training"].to_numpy(),
            label="Self-training",
            linestyle="--",
            marker="s",
        )
        plt.plot(
            results_df["Ensemble with user"].to_numpy(),
            label="Ensemble",
            linestyle="-.",
            marker="^",
        )
        plt.xticks(
            range(len(valid_sample_size_list)),
            [format_func(x, None) for x in valid_sample_size_list],
        )
        plt.xlabel("Number of labeled data")
        plt.ylabel("RMSE")
        plt.title(extract_dataset_name(dataset_name))
        plt.legend()

    ax1 = plt.subplot(n_row, n_col, n_col * n_row)
    combined_results = aggregate_results(results_dict, "regression")
    win_list = combined_results["win_list"]
    win_ratio_list = combined_results["win_ratio_list"]
    valid_count_list = combined_results["valid_count_list"]

    color = "tab:orange"
    ax1.set_xlabel("Number of labeled data")
    ax1.set_ylabel("Win ratio", color=color)
    (line1,) = ax1.plot(
        range(len(n_sample_labeled_data_list)),
        win_ratio_list,
        color=color,
        linestyle="-",
        marker="o",
        label="Ensemble",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(range(len(n_sample_labeled_data_list)))
    ax1.set_xticklabels([format_func(x, None) for x in n_sample_labeled_data_list])

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Dataset count", color=color)
    bar1 = ax2.bar(
        range(len(n_sample_labeled_data_list)),
        valid_count_list,
        color=color,
        alpha=0.5,
        label="Valid",
    )
    bar2 = ax2.bar(
        range(len(n_sample_labeled_data_list)),
        win_list,
        color="tab:blue",
        alpha=0.5,
        label="Win",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    lines = [line1]
    bars = [bar1, bar2]
    labels = [line.get_label() for line in lines] + [bar.get_label() for bar in bars]
    plt.legend(lines + bars, labels, loc="lower left", fontsize=8, framealpha=0.5)
    plt.title("Average performance")

    fig_path = os.path.join(
        fig_dir, f"user_performance_curve_regression_aggregation.pdf"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)


def plot_all_classification_results(results_dict, fig_dir):
    plt.figure(figsize=(13.5, 13.5))
    n_col, n_row = 4, 6
    for i, (dataset_name, result) in results_dict.items():
        valid_sample_size_list = list(result.keys())
        results_df = pd.DataFrame(results_dict[dataset_name]).T

        plt.subplot(n_row, n_col, i + 1)
        plt.plot(
            results_df["Self-training"].to_numpy(),
            label="Self-training",
            linestyle="--",
            marker="s",
        )
        plt.plot(
            results_df["Ensemble with user"].to_numpy(),
            label="Ensemble",
            linestyle="-.",
            marker="^",
        )
        plt.xticks(
            range(len(valid_sample_size_list)),
            [format_func(x, None) for x in valid_sample_size_list],
        )
        plt.xlabel("Number of labeled data")
        plt.ylabel("Accuracy")
        plt.title(extract_dataset_name(dataset_name))
        plt.legend()

    fig_path = os.path.join(
        fig_dir, f"user_performance_curve_classification_all.pdf"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)