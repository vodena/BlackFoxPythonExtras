import numpy as np
import sklearn
from sklearn.metrics import precision_recall_curve, auc as compute_auc
import warnings
from .scaler_extras import rescale_output_data
from .data_extras import prepare_input_data, prepare_output_data
from .encode_data_set import decode_output_data
import tensorflow as tf

def nse_custom_loss_function(y_true, y_pred):
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    nse = 1 - numerator / denominator
    return nse.numpy()

def rmse_peak_sensitive_custom_loss_function(y_true, y_pred):
    q_max = tf.reduce_max(y_true)
    weighted_error = tf.square(y_true - y_pred) * tf.square(y_true / q_max)
    rmse_hf = tf.sqrt(tf.reduce_mean(weighted_error))    
    return rmse_hf.numpy()

def rmse_lows_sensitive_custom_loss_function(y_true, y_pred):
    q_max = tf.reduce_max(y_true)
    weighted_error = tf.square(y_true - y_pred) * tf.square((q_max - y_true) / q_max)
    rmse_hf = tf.sqrt(tf.reduce_mean(weighted_error))    
    return rmse_hf.numpy()

def _calculate_merror(y_true, y_predicted):
    if y_true.shape[1] > 1:
        y_predicted_rounded = np.where(y_predicted == y_predicted.max(axis=1).reshape(len(y_predicted),1), 1, 0)
        A = np.abs(y_predicted_rounded - y_true)
        B = A.sum(axis=1) / 2
        wrong_cases = B.sum(axis=0)
        all_cases = y_true.shape[0]
    else:
        wrong_cases = np.sum(y_true != y_predicted.reshape(len(y_predicted), 1))
        all_cases = y_true.shape[0]
    return wrong_cases / all_cases

def _rmse(y_true, y_pred):
    return sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)

def _mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1) 
    min_value, max_value = np.min(y_true), np.max(y_true)
    abs_percente = np.abs((np.subtract(y_true,y_pred)) / (max_value - min_value))
    return np.mean(abs_percente)

def _pr_auc_score(y_true, y_pred):
    precision_train_metric, recall_train_metric, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = compute_auc(recall_train_metric, precision_train_metric)
    return pr_auc

def get_metric_function_by_name(metric):
    """
    Create metric function by function name

    Suported metrics:

        classification: 'rocauc', 'prauc', 'accuracy', 'precision', 'recall', 'f1score'
        regression : 'mae', 'mape', 'mse', 'rmse', 'r2'
    """
    if metric == "ROC_AUC":
        return sklearn.metrics.roc_auc_score
    elif metric == "PR_AUC":
        return _pr_auc_score
    elif metric == "Precision":
        return sklearn.metrics.precision_score
    elif metric == "Accuracy":
        return sklearn.metrics.accuracy_score
    elif metric == "Recall":
        return sklearn.metrics.recall_score
    elif metric == "F1Score":
        return sklearn.metrics.f1_score
    elif metric == "MAE":
        return sklearn.metrics.mean_absolute_error
    elif metric == "MAPE":
        return _mean_absolute_percentage_error
    elif metric == "MSE":
        return sklearn.metrics.mean_squared_error
    elif metric == "R2":
        return sklearn.metrics.r2_score
    elif metric == "RMSE":
        return _rmse
    elif metric == "NSE":
        return nse_custom_loss_function
    elif metric == "RMSE_PEAK_SENSITIVE":
        return rmse_peak_sensitive_custom_loss_function
    elif metric == "RMSE_LOWS_SENSITIVE":
        return rmse_lows_sensitive_custom_loss_function
    elif metric == "MError":
        return _calculate_merror
    else:
        raise ValueError()

# output is prepared
def calculate_regression_metrics(y, y_pred, metrics = None):
    allowed_metrics = ["MAE", "MAPE", "MSE", "RMSE", "R2", "NSE", "RMSE_PEAK_SENSITIVE", "RMSE_LOWS_SENSITIVE"]

    if metrics is None:
        metrics = allowed_metrics

    calculated_metrics = {}
    for metric in metrics:
        if metric not in allowed_metrics:
            raise Exception("Metric {} isn't supported. Allowed metrics are : {}".format(metric, allowed_metrics))
        
        metric_function = get_metric_function_by_name(metric)
                
        if y is None or y_pred is None:
            calculated_metrics[metric] = -9999
        else:
            calculated_metrics[metric] = metric_function(y, y_pred)
    
    return calculated_metrics

# output is prepared
def calculate_binary_metrics(y, y_pred, metrics = None, threshold = None):
    allowed_metrics = ["ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "F1Score"]
    
    if metrics is None:
        metrics = allowed_metrics
    
    calculated_metrics = {}
    for metric in metrics:
        if metric not in allowed_metrics:
            raise Exception("Metric {} isn't supported. Allowed metrics are : {}".format(metric, allowed_metrics))
        
        metric_function = get_metric_function_by_name(metric)
                
        if y is None or y_pred is None:
            calculated_metrics[metric] = -9999
        else:            
            if metric == "ROC_AUC" or metric == "PR_AUC":
                calculated_metrics[metric] = metric_function(y, y_pred)
            else:
                if threshold is None:
                    y_pred_classes = np.where(y_pred > 0.5, 1, 0)
                    warnings.warn('Threshold not provided. threshold=0.5 will be chosen for {} metric'.format(metric))
                else:
                    y_pred_classes = np.where(y_pred > threshold, 1, 0)
                calculated_metrics[metric] = metric_function(y, y_pred_classes)
    
    return calculated_metrics

# output is prepared - one-hot encoded
def calculate_multiclass_metrics(y, y_pred, metrics = None):
    allowed_metrics = ["ROC_AUC", "Accuracy", "Precision", "Recall", "F1Score", "MError"]
    
    if metrics is None:
        metrics = allowed_metrics

    y_encoded = np.argmax(y, axis=1).reshape(len(y), 1)
    y_pred_encoded = np.argmax(y_pred, axis=1).reshape(len(y), 1)
    
    calculated_metrics = {}
    for metric in metrics:
        if metric not in allowed_metrics:
            raise Exception("Metric {} isn't supported. Allowed metrics are : {}".format(metric, allowed_metrics))
        
        metric_function = get_metric_function_by_name(metric)
                
        if y is None or y_pred is None:
            calculated_metrics[metric] = -9999
        else:
            if metric == "ROC_AUC":
                calculated_metrics[metric] = metric_function(y, y_pred)
            elif metric == "Accuracy" or metric == "MError":
                calculated_metrics[metric] = metric_function(y_encoded, y_pred_encoded)
            else:
                calculated_metrics[metric] = metric_function(y_encoded, y_pred_encoded, average = "micro")
    
    return calculated_metrics