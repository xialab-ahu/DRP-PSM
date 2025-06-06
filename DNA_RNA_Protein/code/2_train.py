
import numpy as np
import pandas as pd

import time
import yaml
import warnings
import os
import argparse
import os.path
from datetime import datetime

import joblib
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from openfe import OpenFE, transform, tree_to_formula

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


callbacks = [early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
callbacks1 = [early_stopping(stopping_rounds=5), lgb.log_evaluation(0)]
callbacks2 = [early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]


def load_data(df_train, y_train):
    """
    Desc:
        load the train feature & label
    Args:
        df_train, train feature
        y_train, train label
    """
    train_x, train_y = df_train, y_train
    X, val_X, y, val_y = train_test_split(
        train_x,
        train_y,
        test_size=0.1,
        random_state=2025,
        stratify=train_y
    )
    lgb_train = lgb.Dataset(X, y)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)
    return lgb_train, lgb_eval


def train_model(df_train, y_train):
    """
    Desc:
        Run lgb with DART in selection parameters
    Args:
        df_train, train feature
        y_train, train label
    """
    lgb_train, lgb_eval = load_data(df_train, y_train)
    gbm = lgb.train(selection_params,
                    lgb_train,
                    num_boost_round=40,
                    valid_sets=[lgb_eval],
                    callbacks=callbacks1
                    )
    return gbm

def autofe(data, openfe_features_path, selectedFea_file, feaName_file):
    """
    Desc:
        run openFE to capture cross-level feature
    Args:
        data, combined DNA-RNA-Protein feature
        openfe_features_path, the save path of generated openfe_feature
        selectedFea_file, the save paht of DART-selected feature after openFE
        feaName_file,
    """
    # Initialize logging
    start_time = time.time()
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting OpenFE feature generation")

    # Separate metadata and features
    train_info = data.iloc[:, :10]
    X_train = data.iloc[:, 10:].astype('float64')
    _, X_test = train_test_split(X_train, test_size=0.1, random_state=4)
    Y_train = np.array(data['label']).reshape(len(data['label']), )

    # generate new features 
    ofe = OpenFE()
    features = ofe.fit(data=X_train, label=Y_train, n_jobs=30)
    # transform feature
    X_train_tr, X_test_tr = transform(X_train, X_test, features, n_jobs=30)
    joblib.dump(features, openfe_features_path)
    total_time = time.time() - start_time
    print(f"**********************[Finished] OpenFE: Completed in {total_time:.2f}s ***********************")
    
    # save OpenFE generated feature name as readable format
    X_train_tr.index = list(range(X_train_tr.shape[0]))
    with open(feaName_file, 'w') as f_write:
        f_write.write('name\n')
        for feature in features:
            f_write.write(tree_to_formula(feature) + '\n')

    # Start DART
    start_time = time.time()
    feature_num = X_train_tr.shape[1]
    print("[INFO] Generated feature num:", feature_num)
    train_round = 1
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting DART feature selection")
    while feature_num > 200:
        gbm = train_model(X_train_tr, Y_train)
        feature_imp = pd.DataFrame({'Value': gbm.feature_importance(), 'Feature': X_train_tr.columns})
        if train_round >= 40:
            break
        train_round += 1
        feature_sum = feature_imp.Value.sum()
        drop_list = feature_imp[feature_imp.Value / feature_sum <= 0].Feature.tolist()
        df = feature_imp
        df["importance"] = feature_imp.Value / feature_sum
        df = df[df["importance"] > 0]
        X_train_tr.drop(drop_list, axis=1, inplace=True)
        feature_num = X_train_tr.shape[1]
    print("[INFO] Selected feature num: ", feature_num)
    total_time = time.time() - start_time
    print(f"**********************[Finished] DART: Completed in {total_time:.2f}s ***********************")
    df = pd.DataFrame(X_train_tr.columns, columns=['feature'])
    df.to_csv(selectedFea_file, index=False)
    return pd.concat([train_info, X_train_tr], axis=1)
    

def step_training_grid(data, model_file, feature_importance_path):
    """
    Desc:
        train DART'selected data from parameter selection using grid search
    Args:
        data, selected features & label
        model_file, saving path for trained model
        feature_importance_path, saving path for feature importance
    """
    print("[INFO] Training for best parameter on DART-selected features...")
    X_train = data.iloc[:, 10:]
    Y_train = data['label']
    X_train = X_train.astype("float64")
    lgb_train = lgb.Dataset(X_train, Y_train, free_raw_data=False)

    # set init params excluding CV params
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.1
    }
    max_auc = float('0')
    best_params = {}

    # Imporve accuracy
    for num_leaves in range(5, 100, 5):
        for max_depth in range(3, 8, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                eval_train_metric=True
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # Avoid over-fitting
    for max_bin in range(5, 256, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                eval_train_metric=True
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['max_bin'] = max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']

    for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    nfold=5,
                    metrics=['auc'],
                    callbacks=callbacks,
                    eval_train_metric=True
                )

                mean_auc = pd.Series(cv_results['valid auc-mean']).max()
                boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']

    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                # verbose_eval=True
                eval_train_metric=True
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']

    for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['auc'],
            callbacks=callbacks,
            eval_train_metric=True
        )

        mean_auc = pd.Series(cv_results['valid auc-mean']).max()
        boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

        if mean_auc >= max_auc:
            max_auc = mean_auc

            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params['min_split_gain'] = best_params['min_split_gain']

    print("**********params*********")
    print(params)
    print("*************************")
    lgb_train, lgb_eval = load_data(X_train, Y_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},  # 二进制对数损失
        'num_leaves': 40,
        'max_depth': 6,
        'max_bin': 255,
        'min_data_in_leaf': 101,
        'learning_rate': 0.01,    
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 45,
        'lambda_l1': 0.001,
        'lambda_l2': 0.4,  # 越小l2正则程度越高
        'min_split_gain': 0.0,
        'verbose': 5,
        'is_unbalance': False
    }
    for key in best_params.keys():
        if key == 'max_depth':
            params[key] = best_params[key]
        elif key == 'max_leaves':
            params[key] = best_params[key]
        else:
            params[key] = best_params[key]
    # train the model from best params
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    callbacks=callbacks2)
    # saving feature importance & model weight
    feature_imp = pd.DataFrame({'Value': gbm.feature_importance(), 'Feature': X_train.columns})
    feature_sum = feature_imp.Value.sum()
    df = feature_imp
    df["importance"] = feature_imp.Value / feature_sum
    df.to_csv(feature_importance_path, index=False)
    joblib.dump(gbm, model_file)


def step_training_instant(data, model_file, feature_importance_path, ready_to_use_param):
    """
    Desc:
        train DART'selected data from Ready-To-Use parameter
    Args:
        data, selected features & label
        model_file, saving path for trained model
        feature_importance_path, saving path for feature importance
        ready_to_use_param, the best param searched by us
    """
    print("[INFO] Retrieve feature importance from DART-selected feature thought LightGBM.")
    start_time = time.time()
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting model training")

    # load data
    X_train = data.iloc[:, 10:] # the first 10 columns are Meta information
    Y_train = data['label']
    X_train = X_train.astype("float64")
    lgb_train = lgb.Dataset(X_train, Y_train, free_raw_data=False)
    lgb_train, lgb_eval = load_data(X_train, Y_train)
    # train model
    gbm = lgb.train(ready_to_use_param,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    callbacks=callbacks2)
    train_time = time.time() - start_time
    # save feature importance
    feature_imp = pd.DataFrame({'Value': gbm.feature_importance(), 'Feature': X_train.columns})
    feature_sum = feature_imp.Value.sum()
    df = feature_imp
    df["importance"] = feature_imp.Value / feature_sum
    df.to_csv(feature_importance_path, index=False)
    # save model
    joblib.dump(gbm, model_file) 
    # Final log
    total_time = time.time() - start_time
    print(
        f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Training completed in {total_time:.2f}s (train: {train_time:.2f}s)\n"
        f"Model saved to: {model_file}\n"
        f"Feature importance saved to: {feature_importance_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="The input train file")
    parser.add_argument("--selectedFea", type=str, help="The save path for DART-selected features")
    parser.add_argument("--openfe_features", type=str, help="The save path for OpenFE-generated features")
    parser.add_argument("--model_file", type=str, help="The save path for output model file")
    parser.add_argument("--feaName_file", type=str, help="The save path for openfe generated feature name")
    parser.add_argument("--lightgbm_importance", type=str, help="The save path for feature importance file")
    parser.add_argument("--from_scratch", action='store_true', help="Run params search from scratch (It may takes a lot of time!)")
    args = parser.parse_args()

    # 1 load selection params for DART process
    with open("/YOUR_PATH/DRP-PSM/DNA_RNA_Protein/code/param_config.yaml") as f: # TODO 🚨 replace the path with your absolute path
        config = yaml.safe_load(f)
    selection_params = config["selection_params"]

    # 2 load data
    data = pd.read_csv(args.train_file)
    colnames = data.columns.tolist()
    colnames = [ ele.replace("[", "") for ele in colnames ]
    colnames = [ ele.replace("]", "") for ele in colnames ]
    colnames = [ ele.replace("(", "") for ele in colnames ]
    colnames = [ ele.replace(")", "") for ele in colnames ]
    colnames = [ ele.replace(" ", "") for ele in colnames ]
    colnames = [ ele.replace(".", "") for ele in colnames ]
    data.columns = colnames
    
    # 3 Generate feature from OpenFE and select feature from DART
    data = autofe(data, args.openfe_features, args.selectedFea,  args.feaName_file)
    
    # 4 training data from dart-selected feature to retrieve feature importance and best param combination
    if args.from_scratch:
        print("[INFO] Training model from SCRATCH to find best param combinations.")
        step_training_grid(data, args.model_file, args.lightgbm_importance)
    else:
        print("[INFO] Training model from Ready-To-Use params.")
        ready_to_use_param = config["hdpsm_params"]
        step_training_instant(data, args.model_file, args.lightgbm_importance, ready_to_use_param)

"""
# A run through command: 
python 2_train.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected.csv \
--selectedFea ../step2-result/train8502_DART_selected_DRP.csv \
--openfe_features ../step2-result/train8502_OpenFE_generated_DRP.csv \
--model_file ../step2-result/train8502_DART_selected_DRP.model \
--feaName_file ../step2-result/train8502_OpenFE_feaName_DRP.csv \
--lightgbm_importance ../step2-result/train8502_DART_selected_DRP_featureImportance.csv
"""