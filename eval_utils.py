import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
import json
from collections.abc import Sequence
from typing import Any, Literal


def unnorm_conf_matr_for_one_case(gt_filepath: str, pred_filepath: str, num_classes: int) -> dict[str, np.ndarray]:
    """
    Compute unnormalized confusion matrices for one test case.
    
    Ground truth and predicted segmentations must be provided as NIfTI files with voxel arrays of the same shape.
    Labels in the arrays must be integers from `0` to `num_classes - 1`.
    
    Args:
        gt_filepath (str): Path to ground truth NIfTI file.
        pred_filepath (str): Path to predicted NIfTI file.
        num_classes (int): Number of classes (including background). Must be at least 2.
        
    Returns:
        np.ndarray: Unnormalized confusion matrix.
    """
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError(f"num_classes must be of type int and at least 2, got {num_classes}.")
    
    gt_nifti = nib.load(gt_filepath)
    pred_nifti = nib.load(pred_filepath)

    # Validate file types and header
    if not isinstance(gt_nifti, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError(f"Ground truth file {gt_filepath} is not a valid NIfTI image.")
    if not isinstance(pred_nifti, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError(f"Prediction file {pred_filepath} is not a valid NIfTI image.")
    if gt_nifti.shape != pred_nifti.shape:
        raise ValueError(f"Ground truth and prediction shapes do not match: {gt_nifti.shape} vs {pred_nifti.shape}.")

    # Load and flatten image data
    gt = gt_nifti.get_fdata(caching='unchanged').flatten()
    pred = pred_nifti.get_fdata(caching='unchanged').flatten()

    # Validate labels in ground truth and prediction
    gt_min, gt_max, pred_min, pred_max = gt.min(), gt.max(), pred.min(), pred.max()
    if gt_min < 0 or pred_min < 0:
        raise ValueError(f"Labels in ground truth or prediction contain negative values. "
                         f"Min label in GT: {gt_min}, Min label in prediction: {pred_min}.")
    if gt_max >= num_classes or pred_max >= num_classes:
        raise ValueError(f"Labels in ground truth or prediction exceed the number of classes: {num_classes}. "
                         f"Max label in GT: {gt_max}, Max label in prediction: {pred_max}.")

    return confusion_matrix(gt, pred, labels=range(num_classes))


def multiclass_to_binary_conf_matr(conf_matr: np.ndarray) -> np.ndarray:
    """
    Convert an unnormalized multiclass confusion matrix to a binary confusion matrix.

    In the binary conversion, class 0 is treated as the background and all other classes
    are merged into a single foreground class (class 1).

    Args:
        conf_matr (numpy.ndarray): An `(n, n)`-shaped unnormalized confusion matrix where index 0 corresponds to the background class.

    Returns:
        numpy.ndarray: A `(2, 2)`-shaped binary confusion matrix
    """
    if conf_matr.ndim != 2 or conf_matr.shape[0] < 2 or conf_matr.shape[0] != conf_matr.shape[1]:
        raise ValueError(f"Confusion matrix have shape (n x n) with n >= 2, got shape {conf_matr.shape}.")

    tn = conf_matr[0, 0]
    fp = conf_matr[0, 1:].sum()
    fn = conf_matr[1:, 0].sum()
    tp = conf_matr[1:, 1:].sum()

    return np.array([[tn, fp],
                     [fn, tp]])


def normalize_conf_matr(conf_matr: np.ndarray, type: Literal['true', 'pred', 'all']) -> np.ndarray:
    """
    Normalize a confusion matrix over true conditions (rows), predicted conditions (columns), or the whole population.

    Args:
        conf_matr (numpy.ndarray): A square matrix where each element `conf_matr[i, j]` is the raw count of instances with true label `i` and predicted label `j`.
        type (str): Normalization type, one of `"true"`, `"pred"`, or `"all"`.
    
    Returns:
        numpy.ndarray: Normalized confusion matrix.
    """
    if conf_matr.ndim != 2 or conf_matr.shape[0] != conf_matr.shape[1]:
        raise ValueError(f"Confusion matrix shape must be (n, n), got shape {conf_matr.shape}.")
    
    
    if type == 'true':
        row_sums = conf_matr.sum(axis=1, keepdims=True)
        return np.divide(conf_matr, row_sums, where=row_sums > 0)
    elif type == 'pred':
        col_sums = conf_matr.sum(axis=0, keepdims=True)
        return np.divide(conf_matr, col_sums, where=col_sums > 0)
    elif type == 'all':
        total_sum = conf_matr.sum()
        return conf_matr / total_sum if total_sum > 0 else conf_matr
    
    else:
        raise ValueError(f"Normalization type must be one of 'true', 'pred', or 'all', got '{type}'.")


def metrics_from_tp_fp_fn(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    Compute segmentation metrics from true positives, true negatives, false positives, and false negatives.
    
    Args:
        tp (int): True Positives
        tn (int): True Negatives
        fp (int): False Positives
        fn (int): False Negatives
    
    Returns:
        dict: Dictionary with precision, recall, Dice, and IoU.
    """
    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Dice": 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else np.nan,
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan,
    }


def metrics_from_unnorm_conf_matr(conf_matr: np.ndarray) -> dict[str, dict[int, dict[str, float]] | dict[str, float]]:
    """
    Compute segmentation metrics from an unnormalized multiclass confusion matrix.

    For multiclass segmentation (one label per voxel), micro-averaged metrics are equivalent to accuracy,
    so only per-class metrics and their unweighted macro-average are computed.

    Args:
        conf_matr (np.ndarray): Square (n x n) unnormalized confusion matrix.

    Returns:
        dict: Dictionary with keys:
            - "per_class": Mapping from class index to a dict of metrics (precision, recall, Dice, IoU).
            - "macro_avg": Macro-averaged metric values computed as the mean across foreground (positive) classes.
    """
    if conf_matr.ndim != 2 or conf_matr.shape[0] != conf_matr.shape[1]:
        raise ValueError(f"Confusion matrix must be square (n x n), got shape {conf_matr.shape}.")
    
    num_classes = conf_matr.shape[0]
    
    TP = np.diag(conf_matr)
    FP = np.sum(conf_matr, axis=0) - TP
    FN = np.sum(conf_matr, axis=1) - TP
    
    per_class = {c: metrics_from_tp_fp_fn(TP[c], FP[c], FN[c]) for c in range(1, num_classes)}
    macro_avg = {metric: np.nanmean([per_class[c][metric] for c in range(1, num_classes)]) 
                 for metric in per_class[1].keys()}
    
    return {"per_class": per_class, "macro_avg": macro_avg}


def metrics_from_unnorm_binary_conf_matr(conf_matr: np.ndarray) -> dict[str, float]:
    """
    Given a binary confusion matrix (for a binary classification task),
    compute precision, recall, Dice, and IoU for the positive class.
    
    Args:
        conf_matr ((2, 2)-shaped array of non-negative integers): An unnormarlized 2x2 confusion matrix with rows/columns for negative class (index 0) and positive class (index 1).
    
    Returns:
        dict: Dictionary with keys "precision", "recall", "Dice", and "IoU" containing the corresponding metrics.
    """
    if conf_matr.shape != (2, 2):
        raise ValueError(f"Cofusion matrix must have shape (2, 2), got shape {conf_matr.shape}.")
    
    tp = conf_matr[1, 1]
    fp = conf_matr[0, 1]
    fn = conf_matr[1, 0]
    return metrics_from_tp_fp_fn(tp, fp, fn)


def save_conf_matr_plot(conf_matr: np.ndarray, labels: Sequence[str], output_path: str, title: str) -> None:
    """
    Save a confusion matrix plot to a file.

    Args:
        conf_matr (np.ndarray): Confusion matrix to plot.
        labels (Sequence[str]): List of label names for the confusion matrix.
        output_path (str): Path to save the confusion matrix image.
        title (str): Title for the confusion matrix plot.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation='vertical')
    plt.title(title)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def evaluate_test_set(gt_dir: str, pred_dir: str, labels: Sequence[str], output_dir: str) -> dict[str, Any]:
    """
    Evaluate multiclass semantic segmentation results for a test set by comparing predicted NIfTI files to ground truth NIfTI files.

    This function computes confusion matrices (unnormalized and normalized over true conditions) and segmentation metrics (precision, recall, Dice, IoU) at various levels:
    - It computes confusion matrices and segmentation metrics per case and at the test set level. Test set-level confusion matrices and metrics are aggregated in two ways: global confusion matrices and metrics are computed by treating the entire test set as a single giant case, while average confusion matrices and metrics are computed by averaging per-case confusion matrices and metrics.
    - Metrics are provided for each class as well as macro-averaged.
    - In addition to multiclass confusion matrices and metrics, this function computes binary confusion matrices and metrics by merging all classes except class 0 (background) into a single foreground class.
    
    The function saves normalized confusion matrix plots and a JSON with detailed results.

    Args:
        gt_dir (str): Path to directory containing ground truth NIfTI files (.nii.gz), one per test case.
        pred_dir (str): Path to directory containing predicted NIfTI files (.nii.gz), matching ground truth filenames.
        labels (Sequence[str]): List of label names, where index corresponds to class id in segmentation mask. Class `labels[0]` is always treated as the background class, although you can use any name for it in `labels`.
        output_dir (str): Directory to save evaluation outputs (global_multiclass_conf_matr.png, global_binary_conf_matr.png, and detailed_eval_results.json).

    Returns:
        dict: Dictionary containing:
            - "global_multiclass_conf_matr": Unnormalized global multiclass confusion matrix (np.ndarray).
            - "global_binary_conf_matr": Unnormalized global binary confusion matrix (np.ndarray).
            - "global_metrics": Dictionary with global metrics for each class, macro average, and merged foreground (dict).
            - "avg_row_norm_multiclass_conf_matr": Average normalized multiclass confusion matrix (np.ndarray).
            - "avg_row_norm_binary_conf_matr": Average normalized binary confusion matrix (np.ndarray).
            - "avg_metrics": Dictionary with average metrics for each class, macro average, and merged foreground (dict).
            - "label_map": Mapping from class index to label name (dict).
            - "case_results": List of per-case confusion matrices and filenames (list of dicts).
            - "saved_files": Paths to saved plot images and JSON (dict).
            - "global_summary_dataframe": Pandas DataFrame with global metrics summary.
            - "avg_summary_dataframe": Pandas DataFrame with average metrics summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get lists of ground truth and prediction files.
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    
    gt_basenames = [os.path.basename(f) for f in gt_files]
    pred_basenames = [os.path.basename(f) for f in pred_files]
    if gt_basenames != pred_basenames:
        raise ValueError("Mismatch between ground truth and prediction filenames.")

    print(f"Found {len(gt_files)} cases. Starting evaluation...")

    num_classes = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}

    global_multiclass_conf_matr = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    avg_row_norm_multiclass_conf_matr = np.zeros((num_classes, num_classes), dtype=np.float64)
    avg_row_norm_binary_conf_matr = np.zeros((2, 2), dtype=np.float64)
    
    # Compute unnormalized and true conditions-normalized per-case confusion matrices
    # and per-case metrics.
    case_results = []
    for gt_file in tqdm(gt_files, desc="Processing cases", unit="cases"):
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)
        
        case_multiclass_cm = unnorm_conf_matr_for_one_case(gt_file, pred_file, num_classes)
        case_binary_cm = multiclass_to_binary_conf_matr(case_multiclass_cm)
        case_row_norm_multiclass_cm = normalize_conf_matr(case_multiclass_cm, type="true")
        case_row_norm_binary_cm = normalize_conf_matr(case_binary_cm, type="true")
        
        global_multiclass_conf_matr += case_multiclass_cm
        avg_row_norm_multiclass_conf_matr += case_row_norm_multiclass_cm
        avg_row_norm_binary_conf_matr += case_row_norm_binary_cm

        case_metrics = metrics_from_unnorm_conf_matr(case_multiclass_cm)
        case_metrics['merged_foreground'] = metrics_from_unnorm_binary_conf_matr(case_binary_cm)
        
        case_results.append({
            "filename": filename,
            "unnorm_multiclass_conf_matr": case_multiclass_cm.tolist(),
            "metrics": case_metrics
        })

    print("Calculating test set-level metrics...")
    # Global confusion matrices
    global_binary_conf_matr = multiclass_to_binary_conf_matr(global_multiclass_conf_matr)

    # Normalized global confusion matrices
    row_norm_global_multiclass_conf_matr = normalize_conf_matr(global_multiclass_conf_matr, type="true")
    row_norm_global_binary_conf_matr = normalize_conf_matr(global_binary_conf_matr, type="true")
    
    # Global metrics
    global_metrics = metrics_from_unnorm_conf_matr(global_multiclass_conf_matr)
    global_metrics['merged_foreground'] = metrics_from_unnorm_binary_conf_matr(global_binary_conf_matr)

    # Average normalized confusion matrices
    avg_row_norm_multiclass_conf_matr /= len(gt_files)
    avg_row_norm_binary_conf_matr /= len(gt_files)

    # Average metrics
    # Calculate average metrics over cases, handling NaNs
    avg_metrics = {"per_class": {}, "merged_foreground": {}, "macro_avg": {}}
    # Per-class metrics (excluding background, i.e., class 0)
    for class_id in range(1, num_classes):
        per_case_vals = {metric: [] for metric in ("precision", "recall", "Dice", "IoU")}
        for case in case_results:
            metrics = case["metrics"]["per_class"].get(class_id, {})
            for metric in per_case_vals:
                val = metrics.get(metric, np.nan)
                per_case_vals[metric].append(val)
        avg_metrics["per_class"][class_id] = {metric: np.nanmean(per_case_vals[metric]) for metric in per_case_vals}
    # Macro average (mean over foreground classes)
    avg_metrics["macro_avg"] = {
        metric: np.nanmean([avg_metrics["per_class"][class_id][metric] for class_id in range(1, num_classes)])
        for metric in ("precision", "recall", "Dice", "IoU")
    }
    # Merged foreground (binary) metrics
    merged_fg_vals = {metric: [] for metric in ("precision", "recall", "Dice", "IoU")}
    for case in case_results:
        metrics = case["metrics"].get("merged_foreground", {})
        for metric in merged_fg_vals:
            val = metrics.get(metric, np.nan)
            merged_fg_vals[metric].append(val)
    avg_metrics["merged_foreground"] = {metric: np.nanmean(merged_fg_vals[metric]) for metric in merged_fg_vals}
    
    # Build summary DataFrame for global metrics.
    global_per_class_metrics = global_metrics["per_class"]
    global_macro_avg = global_metrics["macro_avg"]

    metric_names = list(global_macro_avg.keys())
    global_columns = ["Class ID", "Class Name"] + metric_names
    global_rows = []

    for class_id in sorted(global_per_class_metrics.keys()):
        row = {
            "Class ID": class_id,
            "Class Name": labels[class_id] if class_id < len(labels) else str(class_id)
        }
        for m in metric_names:
            row[m] = global_per_class_metrics[class_id][m]
        global_rows.append(row)

    global_macro_row = {"Class ID": "", "Class Name": "Macro Average"}
    for m in metric_names:
        global_macro_row[m] = global_macro_avg[m]
    global_rows.append(global_macro_row)

    global_fg_row = {"Class ID": "", "Class Name": "Merged Foreground"}
    for m in metric_names:
        global_fg_row[m] = global_metrics['merged_foreground'][m]
    global_rows.append(global_fg_row)

    global_summary_df = pd.DataFrame(global_rows, columns=global_columns)

    # Build summary DataFrame for average metrics (averaged over cases).
    avg_per_class_metrics = avg_metrics["per_class"]
    avg_macro_avg = avg_metrics["macro_avg"]
    avg_columns = ["Class ID", "Class Name"] + metric_names
    avg_rows = []
    for class_id in sorted(avg_per_class_metrics.keys()):
        row = {
            "Class ID": class_id,
            "Class Name": labels[class_id] if class_id < len(labels) else str(class_id)
        }
        for m in metric_names:
            row[m] = avg_per_class_metrics[class_id][m]
        avg_rows.append(row)
    avg_macro_row = {"Class ID": "", "Class Name": "Macro Average"}
    for m in metric_names:
        avg_macro_row[m] = avg_macro_avg[m]
    avg_rows.append(avg_macro_row)
    avg_fg_row = {"Class ID": "", "Class Name": "Merged Foreground"}
    for m in metric_names:
        avg_fg_row[m] = avg_metrics['merged_foreground'][m]
    avg_rows.append(avg_fg_row)
    avg_summary_df = pd.DataFrame(avg_rows, columns=avg_columns)

    print("Saving evaluation results...")
    # Plot and save confusion matrices
    multiclass_conf_matr_img_path = os.path.join(output_dir, "global_multiclass_conf_matr.png")
    save_conf_matr_plot(
        row_norm_global_multiclass_conf_matr,
        labels=labels,
        output_path=multiclass_conf_matr_img_path,
        title="Global Confusion Matrix (Normalized over True Conditions)"
    )

    binary_conf_matr_img_path = os.path.join(output_dir, "global_binary_conf_matr.png")
    save_conf_matr_plot(
        row_norm_global_binary_conf_matr,
        labels=["background", "foreground"],
        output_path=binary_conf_matr_img_path,
        title="Global Merged-Foreground vs Background Confusion Matrix (Normalized over True Conditions)"
    )

    avg_multiclass_conf_matr_img_path = os.path.join(output_dir, "average_multiclass_conf_matr.png")
    save_conf_matr_plot(
        avg_row_norm_multiclass_conf_matr,
        labels=labels,
        output_path=avg_multiclass_conf_matr_img_path,
        title="Average Multiclass Confusion Matrix (Normalized over True Conditions)"
    )
    avg_binary_conf_matr_img_path = os.path.join(output_dir, "average_binary_conf_matr.png")
    save_conf_matr_plot(
        avg_row_norm_binary_conf_matr,
        labels=["background", "foreground"],
        output_path=avg_binary_conf_matr_img_path,
        title="Average Merged-Foreground vs Background Confusion Matrix (Normalized over True Conditions)"
    )

    # Save results to JSON
    saved_files = {}
    saved_files["multiclass_conf_matr_image"] = multiclass_conf_matr_img_path
    saved_files["binary_conf_matr_image"] = binary_conf_matr_img_path
    saved_files["avg_multiclass_conf_matr_image"] = avg_multiclass_conf_matr_img_path
    saved_files["avg_binary_conf_matr_image"] = avg_binary_conf_matr_img_path

    eval_results = {
        "global_multiclass_conf_matr": global_multiclass_conf_matr.tolist(),
        "global_binary_conf_matr": global_binary_conf_matr.tolist(),
        "global_metrics": global_metrics,
        "avg_row_norm_multiclass_conf_matr": avg_row_norm_multiclass_conf_matr.tolist(),
        "avg_row_norm_binary_conf_matr": avg_row_norm_binary_conf_matr.tolist(),
        "avg_metrics": avg_metrics,
        "label_map": label_map,
        "case_results": case_results
    }
    json_out_path = os.path.join(output_dir, "detailed_eval_results.json")
    with open(json_out_path, "w") as json_file:
        json.dump(eval_results, json_file, indent=2)
    saved_files["results_json"] = json_out_path

    print(f"Evaluation complete.")

    return {
        "global_multiclass_conf_matr": global_multiclass_conf_matr,
        "global_binary_conf_matr": global_binary_conf_matr,
        "global_metrics": global_metrics,
        "avg_row_norm_multiclass_conf_matr": avg_row_norm_multiclass_conf_matr,
        "avg_row_norm_binary_conf_matr": avg_row_norm_binary_conf_matr,
        "avg_metrics": avg_metrics,
        "label_map": label_map,
        "case_results": case_results,
        "saved_files": saved_files,
        "global_summary_dataframe": global_summary_df,
        "avg_summary_dataframe": avg_summary_df
    }


def analyze_nnUNet_summary(summary_json_path: str, dataset_json_path: str) -> pd.DataFrame:
    """
    Analyze nnUNet results from summary.json and dataset.json files.

    Args:
        summary_json_path (str): Path to the summary.json file containing per-case evaluation results.
        dataset_json_path (str): Path to the dataset.json file containing class names.

    Returns:
        pd.DataFrame: A DataFrame containing per-class, macro-averaged, and micro-averaged metrics.
    """
    metric_names = ('recall', 'precision', 'Dice', 'IoU')

    with open(summary_json_path, "r") as f:
        summary_json_data = json.load(f)

    with open(dataset_json_path, "r") as f:
        dataset_json_data = json.load(f)

    class_ids = sorted(list(summary_json_data["mean"].keys()))
    class_names = {str(v): k for k, v in dataset_json_data["labels"].items()}  # class_id -> class_name mapping

    # 1. Calculate metrics
    # 1.1. Calculate per-case metrics
    per_case_results = []
    for case in summary_json_data["metric_per_case"]:
        for cl in class_ids:
            tp, fp, fn = [case['metrics'][cl][m] for m in ('TP', 'FP', 'FN')]
            case['metrics'][cl].update(metrics_from_tp_fp_fn(tp, fp, fn))
        per_case_results.append({
            "case_id": os.path.basename(case["prediction_file"]),
            "per_class_metrics": case['metrics'],
        })

    # 1.2. Calculate per-class metrics by averaging per-case metrics
    class_results = {
        cl: {
            metric_name: np.nanmean([case["per_class_metrics"][cl][metric_name] for case in per_case_results])
            for metric_name in metric_names
        }
        for cl in class_ids
    }

    # 1.3. Average the per-class metrics (macro average)
    macro_avg = {
        metric_name: np.nanmean([class_results[cl][metric_name] for cl in class_ids])
        for metric_name in metric_names
    }

    # 2. Create DataFrame
    # Helper to format a single metric record
    def format_record(class_id: str, class_name: str, metrics: dict) -> dict:
        return {
            "Class ID": class_id,
            "Class Name": class_name,
            "Recall": metrics["recall"],
            "Precision": metrics["precision"],
            "Dice": metrics["Dice"],
            "IoU": metrics["IoU"],
        }

    # Create DataFrame
    records = [format_record(cl, class_names[cl], class_results[cl]) for cl in class_ids]
    records.append(format_record("macro", "Macro Average", macro_avg))

    return pd.DataFrame(records)