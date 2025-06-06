"""
@description: This script runs model training solely on protein structure features for ablation experiments.
"""
import argparse
import os.path
import joblib
import numpy as np
import pandas as pd

from lightgbm import early_stopping
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import lightgbm as lgb
import warnings

import time
import yaml
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
callbacks = [early_stopping(stopping_rounds = 10)]
callbacks1 = [early_stopping(stopping_rounds = 5)]
callbacks2 = [early_stopping(stopping_rounds = 50)]


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
        test_size=0.2,
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


def select_fea(data, selectedFea_file):
    """
    Desc:
        Train model to select DNA features through DART (Dropouts meet Multiple Additive Regression Trees).
        Iteratively removes non-important features until <=200 features remain or max rounds (40) reached.
    Args:
        data: DataFrame containing training data (first 10 columns are metadata, rest are features)
        selectedFea_file: Path to save the selected features (CSV format)
    Returns:
        DataFrame: Original metadata columns + selected feature columns
    """
    print("[INFO] Running DART to filter out NON-important features...")
    # Initialize logging
    start_time = time.time()
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting DART feature selection")
    
    # Separate metadata and features
    train_info = data.iloc[:, :10]  # First 10 columns assumed to be metadata
    X_train = data.iloc[:, 10:].astype('float64')
    Y_train = data["label"]
    
    # Reset index for safety
    X_train.index = range(len(X_train))
    initial_features = X_train.shape[1]
    print(f"[INFO] Initial features: {initial_features:,}")
    
    # DART feature selection loop
    train_round = 1
    while X_train.shape[1] > 200:
        round_start = time.time()
        
        # Train model and get feature importance
        gbm = train_model(X_train, Y_train)
        feature_imp = pd.DataFrame({
            'Feature': X_train.columns,
            'Value': gbm.feature_importance(),
            'Importance': gbm.feature_importance() / gbm.feature_importance().sum()
        })
        
        # Identify features to drop (zero importance)
        drop_list = feature_imp[feature_imp['Importance'] <= 0]['Feature'].tolist()
        remaining_features = feature_imp[feature_imp['Importance'] > 0]
        
        # Logging for current round
        print(
            f"[ROUND {train_round}] "
            f"Dropped: {len(drop_list):,} features | "
            f"Remaining: {remaining_features.shape[0]:,} | "
            f"Time: {time.time() - round_start:.2f}s"
        )
        
        # Update features
        X_train.drop(columns=drop_list, inplace=True)
        train_round += 1
        
        # Early stopping
        if train_round >= 40:
            print("[WARNING] Max rounds (40) reached before reaching <=200 features")
            break
    
    # Final output
    final_feature_count = X_train.shape[1]
    total_time = time.time() - start_time
    print(
        f"\n[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Completed in {total_time:.2f}s ({train_round} rounds)\n"
        f"Initial features: {initial_features:,} → Final features(without meta headers): {final_feature_count:,}"
    )
    
    # Save and return results
    pd.DataFrame({'feature': X_train.columns}).to_csv(selectedFea_file, index=False)
    return pd.concat([train_info, X_train], axis=1)


def step_training_rdcv(data, model_file, feature_importance_path):
    """
    Desc:
        train protein structural feature from Ready-To-Use parameter using RandomSearchCV
    Args:
        data, selected features & label
        model_file, saving path for trained model
        feature_importance_path, saving path for feature importance
    """
    X = data.iloc[:, 10:]
    Y = data['label']
    X = X.astype("float64")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=2024)

    param_dist = {
        'num_leaves': range(5, 100, 5),
        'max_depth': range(3, 8, 1),
        'max_bin': range(5, 256, 10),
        'min_data_in_leaf': range(1, 102, 10),
        'learning_rate': [0.001, 0.004, 0.007, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4,0.5],
        'feature_fraction': np.linspace(0.6, 1.0, 5),  
        'bagging_fraction': np.linspace(0.6, 1.0, 5),
        'bagging_freq': range(0, 50, 5),
        'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda_l2': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_split_gain': np.linspace(0.0, 1.0, 11)
    }

    fixed_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'verbose': -1  
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    
    # Init LGBMClassifier
    lgb_clf = lgb.LGBMClassifier(**fixed_params)
    # Use RandomizedSearchCV for random param search & cross validation
    random_search = RandomizedSearchCV(
        estimator=lgb_clf,
        param_distributions=param_dist,
        n_iter=5000, 
        scoring='roc_auc',  
        cv=cv,  
        n_jobs=-1, 
        verbose=1,  
        random_state=2024,
    )
    
    random_search.fit(X, Y)
    
    # Output best param combination
    print("**********Best Params*********")
    print(random_search.best_params_)
    print("*************************")
    print(f"Best AUC: {random_search.best_score_:.4f}")
    
    lgb_train, lgb_eval = load_data(X, Y) 
    
    # Train final model use best param combination
    best_params = random_search.best_params_
    final_params = {**fixed_params, **best_params}
    final_params['metric'] = ['binary_logloss', 'auc'] 

    with open(model_file.replace(".pkl", "_final_params.yaml"), "w") as f:
        yaml.dump(final_params, f)

    gbm = lgb.train(
        final_params,
        lgb_train,
        num_boost_round=1000,   
        valid_sets=lgb_eval,
        callbacks=callbacks2
    )
    
    feature_imp = pd.DataFrame({'Value': gbm.feature_importance(), 'Feature': X.columns})
    feature_sum = feature_imp.Value.sum()
    df = feature_imp
    df["importance"] = feature_imp.Value / feature_sum
    df.to_csv(feature_importance_path, index=False)
     
    # Saving model
    joblib.dump(gbm, model_file) 


def step_training_instant(data, model_file, feature_importance_path, ready_to_use_param):
    """
    Desc:
        train protein structural feature from Ready-To-Use parameter
    Args:
        data, selected features & label
        model_file, saving path for trained model
        feature_importance_path, saving path for feature importance
        ready_to_use_param, the best param searched by us
    """
    print("[INFO] Train protein structural feature thought LightGBM.")
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
    parser.add_argument("--model_file", type=str, help="The save path for output model file")
    parser.add_argument("--lightgbm_importance", type=str, help="The save path for feature importance file")
    parser.add_argument("--from_scratch", action='store_true', help="Run params search from scratch (It may takes a lot of time!)")
    parser.add_argument("--sequence_len", choices=['7aa', '17aa', '27aa', '37aa', '47aa', '57aa', '67aa'], help="The length of Protein sequence for current job")
    args = parser.parse_args()

    # 1 load data
    data = pd.read_csv(args.train_file)
    colnames = data.columns.tolist()
    colnames = [ ele.replace("[", "") for ele in colnames ]
    colnames = [ ele.replace("]", "") for ele in colnames ]
    data.columns = colnames

    # 2 get parameter
    with open("/YOUR_PATH/DRP-PSM/Protein_features/code/params_config.yaml") as f:  # TODO 🚨 Replace this path as your absolute path
        config = yaml.safe_load(f)
    selection_params = config["selection_params"]

    # 3 Select feature from dart
    data = select_fea(data, args.selectedFea) 
    print(data.shape) 

    # 4 Train model
    if args.from_scratch:
        print("[INFO] Training model from SCRATCH to find best param combinations.")
        step_training_rdcv(data, args.model_file, args.lightgbm_importance)
    else:
        print("[INFO] Training model from Ready-To-Use params.")
        ready_to_use_param = config[args.sequence_len]
        step_training_instant(data, args.model_file, args.lightgbm_importance, ready_to_use_param)

"""
# A run-through command

python 5_train_solely_on_structure.py \
--train_file ../protein_structure/train8502_17aa_proteinfea.csv \
--selectedFea ../result_structure/reproduce_train8502_protein_17aa_DART_selected_features.csv \
--lightgbm_importance ../result_structure/reproduce_train8502_17aa_selected_importance.csv \
--model_file ../result_structure/reproduce_train8502_17aa_selected.model \
--sequence_len 17aa
"""