"""
@desciption: predict on test set based on RNA / RNA structural data.
"""
import argparse

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, \
    matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def predict(importance_file, test_file, model_file, ablation_name):
    """
    Desc:
        predict result on test dataset
    Args:
        importance_file, the selected feature file
        test_file, test file with all original feature
        model_file, the trained model_file on sole structural/all features
        ablation_name, the ablation name of current test
    """
    # 1 load feature subset
    df_import = pd.read_csv(importance_file)
    fea_list = df_import["Feature"].tolist()

    # 2 load test file
    test = pd.read_csv(test_file, low_memory = False)
    
    # 3 retrieve feature subset
    test_info = test.iloc[:, :10]
    X_test = test[fea_list].astype('float64')

    # 4 load model
    model = joblib.load(model_file)

    # 5 predict & save result
    test_pred = model.predict(X_test)
    return pd.concat([test_info, X_test, pd.DataFrame(test_pred, columns=[ablation_name])], axis=1)


def calculate_metrics(df_pred, save_path, ablation_name):
    """
    Desc:
        calculdate binary classification result
    Args:
        df_pred, predicted result from predict()
        save_path, the output path for metrics
        ablation_name, the ablation name of current test
    """
    df_pred["y_pred_classes"] = np.where(df_pred[ablation_name] > 0.6, 1, 0) # use 0.6 as a cutoff
    y_label = df_pred["label"].tolist()
    y_pred = df_pred[ablation_name].tolist()
    y_pred_classes = df_pred["y_pred_classes"].tolist()

    classification_metrics = classification_report(y_label, y_pred_classes)
    print(classification_metrics)

    cm = confusion_matrix(y_label, y_pred_classes)
    tn, fp, fn, tp = cm.ravel()

    # ACC
    acc = accuracy_score(y_label, y_pred_classes)
    # SEN
    sen = recall_score(y_label, y_pred_classes)
    # SPE
    spe = tn / (tn + fp)
    # PRE
    pre = precision_score(y_label, y_pred_classes)
    # F1
    f1 = f1_score(y_label, y_pred_classes)
    #  MCC
    mcc = matthews_corrcoef(y_label, y_pred_classes)

    print("ACC = %.4f" % acc)
    print("SEN = %.4f" % sen)
    print("SPE = %.4f" % spe)
    print("PRE = %.4f" % pre)
    print("F1 = %.4f" % f1)
    print("MCC = %.4f" % mcc)

    auc_score = roc_auc_score(y_label, y_pred)
    precision, recall, _ = precision_recall_curve(y_label, y_pred)
    aupr = auc(recall, precision)
    print("\nAUC = %.4f" % auc_score, "    AUPR =%.4f" % aupr)

    df = pd.DataFrame(
        {
            "Accuracy": [acc],
            "Sensitivity": [sen],
            "Specificity": [spe],
            "Precision": [pre],
            "F1-score": [f1],
            "MCC": [mcc],
            "AUC": [auc_score],
            "AUPR": [aupr],
        })
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, help="The test dataset with all original feature")
    parser.add_argument("--importance_file", type=str, help="The selected feature file")
    parser.add_argument("--model_file", type=str, help="The model trained on selected feature")
    parser.add_argument("--outfile", type=str, help="The output path for predicted result")
    parser.add_argument("--outfile_metrics", type=str, help="The output path for predicted metrics")
    parser.add_argument("--ablation_name", type=str, help="The ablation name for current test")
    args = parser.parse_args()

    # 1 retrieve predict result
    df_pred = predict(args.importance_file, args.test_file, args.model_file, args.ablation_name)
    df_pred.to_csv(args.outfile, index=False)

    # 2 calculate metrics
    calculate_metrics(df_pred, args.outfile_metrics, args.ablation_name)

"""
# All feature
python 2_predict_RNA.py \
--test_file ../RNA_seq_structure/test816_RNA_141bp_allfea.csv \
--importance_file ../result_seq_structure/reproduce_train8502_RNAfea_141bp_importance.csv \
--model_file ../result_seq_structure/reproduce_train8502_RNAfea_141bp.model \
--outfile ../result_seq_structure/reproduce_test816_141bp_predicted.csv \
--outfile_metrics ../result_seq_structure/reproduce_test816_141bp_predicted_metrics.csv \
--ablation_name RNA

# Structural feature
python 2_predict_RNA.py \
--test_file ../RNA_structure/test816_RNA_141bp_structurefea.csv \
--importance_file ../result_structure/reproduce_train8502_141bp_selected_importance.csv \
--model_file ../result_structure/reproduce_train8502_141bp_selected.model \
--outfile ../result_structure/reproduce_test816_141bp_predicted.csv \
--outfile_metrics ../result_structure/reproduce_test816_141bp_predicted_metrics.csv \
--ablation_name RNA_structural
"""