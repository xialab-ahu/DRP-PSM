"""
@description: This script uses DART-selected feature and best param to perform 5-fold validation
"""
import numpy as np
import pandas as pd
import yaml
import argparse

import time
from datetime import datetime
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score


def fold(X_train, y_train, importance_file, png_file, params, save_prefix):
    """
    Desc:
        5-fold on DART-selected features & best parameters
    Args:
        X_train, train feature data
        y_train, train label
        importance_file, feature importance file from lightGBM
        png_file, save path for 5-fold validation result
        params, best params from yaml file to train lgbm
        save_prefix, the prefix when saving files
    """
    print("[INFO] Start k-fold to retrieve prediction result from specific sequence length and DART-selected features.")
    start_time = time.time()
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting model training")

    # Use DART-selected feature to refine orginal feature file
    colnames = X_train.columns.tolist()
    colnames = [ ele.replace("[", "") for ele in colnames ]
    colnames = [ ele.replace("]", "") for ele in colnames ]
    X_train.columns = colnames 
    df_importance = pd.read_csv(importance_file)
    fea_list = df_importance["Feature"].tolist()
    X_train = X_train[fea_list]
    
    # Split into 5-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=2025)

    # Init result list
    auroc_scores = []
    aupr_scores = []

    # 5-fold training
    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        # Pack data
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

        # Train data
        model = lgb.train(params, train_data, valid_sets=[train_data, val_data], num_boost_round=1000, callbacks=[early_stopping(stopping_rounds=50)])
        
        # Save model result for this fold
        feature_imp = pd.DataFrame({'Value': model.feature_importance(), 'Feature': X_train.columns})
        feature_sum = feature_imp.Value.sum()
        df = feature_imp
        df["importance"] = feature_imp.Value / feature_sum
        df.to_csv(f"{save_prefix}_fold{fold}.csv", index=False)

        # Predict result
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)

        # Calculate metrics
        auroc = roc_auc_score(y_val_fold, y_pred)
        aupr = average_precision_score(y_val_fold, y_pred)
        auroc_scores.append(auroc)
        aupr_scores.append(aupr)

    # Calculdate average result 
    avg_auroc = np.mean(auroc_scores)
    avg_aupr = np.mean(aupr_scores)

    total_time = time.time() - start_time
    print(
        f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Training completed in {total_time:.2f}s (train: {total_time:.2f}s)\n"
        f"Feature importance saved to: {save_prefix}"
    )

    # Plot result
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), auroc_scores, label='AUROC')
    plt.plot(range(1, 6), aupr_scores, label='AUPR')
    plt.axhline(avg_auroc, color='r', linestyle='--', label=f'Average AUROC: {avg_auroc:.4f}')
    plt.axhline(avg_aupr, color='b', linestyle='--', label=f'Average AUPR: {avg_aupr:.4f}')
    plt.xlabel('Fold', size=12)
    plt.ylabel('Score', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title('Five-fold Cross Validation Results', size=16)
    plt.legend()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    print("AUC:", avg_auroc)
    print("AUPR: ", avg_aupr)
    plt.savefig(png_file)
    # plt.show()

    # Save result
    df_metrics = pd.DataFrame({'folds': list(range(1, 6)), 'auc': auroc_scores, 'aupr': aupr_scores})
    df_metrics.to_csv(f"{save_prefix}_allfold_metrics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="file stores train feature & label")
    parser.add_argument("--png_file", type=str, help="saved path for 5-fold result")
    parser.add_argument("--prefix", type=str, help="The save path for saving files")
    parser.add_argument("--importance_file", help="lightGBM feature importance file generated from 1_selection.py after best params.")
    parser.add_argument("--sequence_len", choices=['21bp', '51bp', '81bp', '111bp', '141bp', '171bp', '201bp'], help="The length of DNA sequence for current job")

    # load data
    args = parser.parse_args()        
    df_train = pd.read_csv(args.train_file, sep=",")
    X_train = df_train.iloc[:, 10:]
    y_train = df_train["label"]

    # load params
    with open("/YOUR_PATH/DRP-PSM/DNA_features/code/params_config.yaml") as f: # TODO 🚨 Replace this path as your absolute path
        config = yaml.safe_load(f)
    params = config[args.sequence_len]

    # 5-fold
    fold(X_train, y_train, args.importance_file, args.png_file, params, args.prefix)


"""
# Run through example
python 3-foldValidation.py \
--train_file ../allfea/train8502_DNAfea_21bp.csv \
--importance_file  ../result/reproduce_train8502_DNAfea_21bp_importance.csv \
--png_file ../fig_5foldValidation/reproduce_train8502_DNAfea_21bp_5fold.png \
--sequence_len 21bp \
--prefix ../fig_5foldValidation/train8502_DNAfea_21bp_DART_selected_features_importance_
"""





