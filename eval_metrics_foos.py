import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


def classification_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_score_macro = f1_score(y_true, y_pred, average='macro')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_score_micro = f1_score(y_true, y_pred, average='micro')

    print("confusion: \n", conf_matrix)
    print("accuracy: ", accuracy)
    print("precision_macro: ", precision_macro)
    print("recall_macro: ", recall_macro)
    print("f1_score_macro: ", f1_score_macro)
    print("precision_micro: ", precision_micro)
    print("recall_micro: ", recall_micro)
    print("f1_score_micro: ", f1_score_micro)


def rmse_func(truth, pred):
	rmse = np.sqrt(mean_squared_error(truth, pred))
	return round(rmse, 5)


def mae_func(truth, pred):
	mae = mean_absolute_error(truth, pred)
	return round(mae, 5)




def multi_class_confusion_matrix(true_labels, predicted_labels):
    """
    Create a multi-class confusion matrix from two columns of a Pandas DataFrame.

    Parameters:
        true_labels (pandas.Series): Pandas Series containing true labels.
        predicted_labels (pandas.Series): Pandas Series containing predicted labels.

    Returns:
        pandas.DataFrame: Multi-class confusion matrix.
    """
    unique_labels = sorted(set(true_labels) | set(predicted_labels))
    confusion_matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix.loc[true_label, predicted_label] += 1

    return confusion_matrix


def macro_average_sensitivity(conf_matrix):
    """
    Calculate sensitivity (recall) using macro-averaging based on a multi-class confusion matrix.

    Parameters:
        conf_matrix (pandas.DataFrame): Multi-class confusion matrix.

    Returns:
        float: Macro-averaged sensitivity.
    """
    sensitivities = []
    for label in conf_matrix.index:
        true_positives = conf_matrix.loc[label, label]
        actual_positives = conf_matrix.loc[label].sum()
        sensitivity = true_positives / actual_positives if actual_positives != 0 else 0
        sensitivities.append(sensitivity)

    macro_avg_sensitivity = sum(sensitivities) / len(sensitivities)
    return macro_avg_sensitivity


def macro_average_precision(conf_matrix):
    """
    Calculate precision using macro-averaging based on a multi-class confusion matrix.

    Parameters:
        conf_matrix (pandas.DataFrame): Multi-class confusion matrix.

    Returns:
        float: Macro-averaged precision.
    """
    total_classes = len(conf_matrix)
    class_precisions = []
    
    for i in range(total_classes):
        true_positive = conf_matrix.iloc[i, i]
        false_positive = conf_matrix.iloc[:, i].sum() - true_positive
        false_negative = conf_matrix.iloc[i, :].sum() - true_positive
        
        class_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        class_precisions.append(class_precision)
    
    macro_avg_precision = sum(class_precisions) / total_classes
    return macro_avg_precision


def micro_average_sensitivity(conf_matrix):
    """
    Calculate sensitivity (recall) using micro-averaging based on a multi-class confusion matrix.

    Parameters:
        conf_matrix (pandas.DataFrame): Multi-class confusion matrix.

    Returns:
        float: Micro-averaged sensitivity.
    """
    true_positives = conf_matrix.values.diagonal().sum()
    total_positives = conf_matrix.values.sum()
    micro_avg_sensitivity = true_positives / total_positives if total_positives != 0 else 0
    return micro_avg_sensitivity


def class_metrics(conf_matrix):
    """
    Calculate precision, recall, true positives, false positives, false negatives,
    and true negatives for each class based on a multi-class confusion matrix.

    Parameters:
        conf_matrix (pandas.DataFrame): Multi-class confusion matrix.

    Returns:
        pandas.DataFrame: DataFrame containing class-wise metrics.
    """
    class_names = conf_matrix.index
    metrics = []

    for class_name in class_names:
        true_positive = conf_matrix.loc[class_name, class_name]
        false_positive = conf_matrix.loc[:, class_name].sum() - true_positive
        false_negative = conf_matrix.loc[class_name, :].sum() - true_positive
        true_negative = conf_matrix.values.sum() - (true_positive + false_positive + false_negative)
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        
        metrics.append([class_name, true_positive, false_positive, false_negative, true_negative, precision, recall])

    df_metrics = pd.DataFrame(metrics, columns=['Class', 'True Positive', 'False Positive', 'False Negative', 'True Negative', 'Precision', 'Recall'])
    return df_metrics


# # Example usage:
# sentiment_df = pd.read_excel('ing_data_1.xlsx')

# true_labels = (sentiment_df['c_score_2'] / 2).round().clip(1,5)
# predicted_labels = (sentiment_df['c_score_3'] / 2).round().clip(1,5)

# conf_matrix = multi_class_confusion_matrix(true_labels, predicted_labels)


# print("Multi-class Confusion Matrix:")
# print(conf_matrix)

# print(class_metrics(conf_matrix))

# print("Macro-Averaged Sensitivity:", macro_average_sensitivity(conf_matrix))
# print("Macro-Averaged Precision:", macro_average_precision(conf_matrix))
# print("Micro-Averaged Sensitivity:", micro_average_sensitivity(conf_matrix))
