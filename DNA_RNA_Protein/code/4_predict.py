"""
@description: This script helps to predict result for independent test dataset
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import os
import time
from openfe import transform
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, \
    matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def predict(train_file, test_file, autoFE_features, selection, model_file):
    """
    Desc:
        predict result on test dataset
    Args:
        train_file, original train file
        test_file, original test file
        autoFE_features, the generatd autoFE_features
        selection, the DART-selected feature
        model_file, the trained model file
    """
    # 1 Load data
    train = pd.read_csv(train_file, low_memory=False)
    test = pd.read_csv(test_file, low_memory = False)
    
    colnames = train.columns.tolist()
    colnames = [ ele.replace("[", "") for ele in colnames ]
    colnames = [ ele.replace("]", "") for ele in colnames ]
    colnames = [ ele.replace("(", "") for ele in colnames ]
    colnames = [ ele.replace(")", "") for ele in colnames ]
    colnames = [ ele.replace(" ", "") for ele in colnames ]
    colnames = [ ele.replace(".", "") for ele in colnames ]
    train.columns = colnames
    
    colnames = test.columns.tolist()
    colnames = [ ele.replace("[", "") for ele in colnames ]
    colnames = [ ele.replace("]", "") for ele in colnames ]
    colnames = [ ele.replace("(", "") for ele in colnames ]
    colnames = [ ele.replace(")", "") for ele in colnames ]
    colnames = [ ele.replace(" ", "") for ele in colnames ]
    colnames = [ ele.replace(".", "") for ele in colnames ]
    test.columns = colnames
    
    test_info = test.iloc[:, :10]
    features = joblib.load(autoFE_features)
    
    X_train = train.iloc[:, 10:].astype('float64')
    X_test = test.iloc[:, 10:].astype('float64')

    print('---' + time.asctime(time.localtime(time.time())) + '--- transforming dataset\n')
    # 2 transform test data based on orginal train & openFE train
    _, X_test_tr = transform(X_train, X_test, features, n_jobs=30)
    feature_list_final = pd.read_csv(selection).feature.tolist()
    print(feature_list_final)
    # select test data based on dart result 
    X_test_filtered = X_test_tr[feature_list_final].astype('float64')
    print(X_test_filtered)

    print('---' + time.asctime(time.localtime(time.time())) + '--- predicting\n')
    # 3 load model weight and retrieve predict result
    model = joblib.load(model_file)
    test_pred = model.predict(X_test_filtered)
    return pd.concat([test_info, X_test_filtered, pd.DataFrame(test_pred, columns=['HDPSM'])], axis=1)


def calculate_metrics(df_pred, save_path):
    """
    Desc:
        calculdate binary classification result
    Args:
        df_pred, predicted result from predict()
        save_path, the output path for metrics
    """
    name = "HDPSM"
    df_pred["y_pred_classes"] = np.where(df_pred[name] > 0.6, 1, 0) # use 0.6 as a cutoff
    y_label = df_pred["label"].tolist()
    y_pred = df_pred[name].tolist()
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
    parser.add_argument("--train_file", type=str, help="The input train file.")
    parser.add_argument("--test_file", type=str, help="The input test file.")
    parser.add_argument("--openfe_features", type=str, help="The input OpenFE generated train feature file.")
    parser.add_argument("--selectedFea", type=str, help="The input DART selected train feature file.")
    parser.add_argument("--model_file", type=str, help="The input trained model file.")
    parser.add_argument("--outfile", type=str, help="The output path for predicted result.")
    parser.add_argument("--outfile_metrics", type=str, help="The output path for predicted metrics.")
    args = parser.parse_args()

    # 1 retrieve predict result
    df_pred = predict(args.train_file, args.test_file, args.openfe_features, args.selectedFea, args.model_file)
    df_pred.to_csv(args.outfile, index=False)
    # 2 calculate metrics
    calculate_metrics(df_pred, args.outfile_metrics)


"""
# Imputed
python 4_predict_HDPSM.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected_imputed.csv \
--test_file ../step1-result/test816_DNA_RNA_Protein_selected_imputed.csv \
--openfe_features ../step2-result/train8502_OpenFE_generated_DRP.csv \
--selectedFea ../step2-result/train8502_DART_selected_DRP.csv \
--model_file ../step2-result/train8502_DART_selected_DRP.model \
--outfile ../step2-result/test816_predicted_imputed.csv \
--outfile_metrics ../step2-result/test816_predicted_metrics.csv
"""

