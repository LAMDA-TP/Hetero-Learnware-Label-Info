import numpy as np
from scipy import stats

def calculate_mean_results(performance_list):
    performance_list = np.mean(performance_list, axis=0)
    mean = np.mean(performance_list)
    std = np.std(performance_list)
    return mean, std

def complete_y_user(y_user, model_classes, all_classes):
    y_user_complete = np.zeros((y_user.shape[0], len(all_classes)))
    for i, model_class in enumerate(model_classes):
        y_user_complete[:, all_classes.index(model_class)] = y_user[:, i]
    return y_user_complete

def fill_nan_with_exception_early(data, method="mean", fill_value=0):
    for col in range(data.shape[1]):
        if np.isnan(data[:, col]).all():
            raise ValueError(f"Column {col} full of NaN, failed to fill")

    filled_data = np.copy(data)

    if method == "fixed":
        filled_data = np.nan_to_num(filled_data, nan=fill_value)
    elif method in ["mean", "median", "mode"]:
        for col in range(data.shape[1]):
            if method == "mean":
                replace_value = np.nanmean(data[:, col])
            elif method == "median":
                replace_value = np.nanmedian(data[:, col])
            elif method == "mode":
                replace_value, _ = stats.mode(data[:, col], nan_policy="omit")
                replace_value = (
                    replace_value[0] if replace_value.size > 0 else np.nan
                )

            mask = np.isnan(data[:, col])
            filled_data[mask, col] = replace_value

    return filled_data