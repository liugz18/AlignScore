import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np

def Plot_Analysis(data, criterion="align_score"):
    # Step 6: Calculate ROC-AUC and plot ROC curve
    fprs, tprs, thresholds = roc_curve(data['detection_label'], data[criterion].apply(lambda x:-x))
    roc_auc = roc_auc_score(data['detection_label'], data[criterion].apply(lambda x:-x))

    # Plot ROC curve
    plt.plot(fprs, tprs, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC on {criterion} (Positive:Hall detected)')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # Assuming you have calculated fpr, tpr, thresholds
    # print(thresholds)
    # Calculate F1 scores for each threshold
    f1_scores = [f1_score(data['detection_label'], data[criterion].apply(lambda x: -x) >= threshold) for threshold in thresholds]
    precision_scores = [precision_score(data['detection_label'], data[criterion].apply(lambda x: -x) >= threshold) for threshold in thresholds]
    recall_scores = [recall_score(data['detection_label'], data[criterion].apply(lambda x: -x) >= threshold) for threshold in thresholds]
    accuracy_scores = [accuracy_score(data['detection_label'], data[criterion].apply(lambda x: -x) >= threshold) for threshold in thresholds]

    # for threshold, f1, tpr, fpr, prec, rec, acc in zip(thresholds, f1_scores, tprs, fprs, precision_scores, recall_scores, accuracy_scores):
    #     print(f'Threshold: {-1 * threshold}, TPR: {tpr}, FPR: {fpr}, F1 score: {f1}, Prec: {prec}, Rec: {rec}, Acc: {acc}')

    # Find the best threshold that maximizes the F1 score
    best_threshold = -1 * thresholds[np.argmax(f1_scores)]

    # Find the best threshold that maximizes precision
    best_precision_threshold = -1 * thresholds[np.argmax(precision_scores)]

    # Find the best threshold that maximizes recall
    best_recall_threshold = -1 * thresholds[np.argmax(recall_scores)]

    # Find the best threshold that maximizes accuracy
    best_accuracy_threshold = -1 * thresholds[np.argmax(accuracy_scores)]

    print("Best F1 Threshold:", best_threshold)
    print("Best Precision Threshold:", best_precision_threshold)
    print("Best Recall Threshold:", best_recall_threshold)
    print("Best Accuracy Threshold:", best_accuracy_threshold)

    # Print metrics for each threshold on validation set
    thresholds = -1 * thresholds
    thresholds[0] = -0.01
    for threshold, f1, tpr, fpr, prec, rec, acc in zip(thresholds, f1_scores, tprs, fprs, precision_scores, recall_scores, accuracy_scores):
        if threshold in [best_threshold, best_precision_threshold, best_recall_threshold, best_accuracy_threshold] or abs(threshold - 0.58) < 0.01:
            print(f'Threshold: {threshold}, TPR: {tpr}, FPR: {fpr}, F1 score: {f1}, Prec: {prec}, Rec: {rec}, Acc: {acc}')

    # Plot metrics as thresholds change
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, accuracy_scores, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Score')
    plt.title('Metrics vs. Threshold on Test Set')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Read the CSV file into a pandas DataFrame
    csv_file_path = "train_data_with_Alignscore.csv"#"test.csv" #
    df = pd.read_csv(csv_file_path)
    df['align_score'] = df['align_score'].apply(lambda x: float(x[1:-1]))
    # print(type(df['score'][0]))
    # Call the Plot_Analysis() function and pass the DataFrame as an argument
    Plot_Analysis(df)
