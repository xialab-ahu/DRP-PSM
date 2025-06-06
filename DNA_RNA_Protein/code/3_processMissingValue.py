"""
@description: This script helps to impute missing values from train & test data
"""
import time
import argparse
import pandas as pd
from datetime import datetime
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def processMissing(train_file, test_file, out_train, out_test):
    """
    Desc:
        Impute missing value based Bayesian Iterate model
    Args:
        train_file, train file
        test_file, test file
    """
    print("[INFO] Start imputing data...")
    start_time = time.time()
    print(f"\n[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting model training")

    # load train data
    df_train = pd.read_csv(train_file)
    listFea_train = df_train.columns.tolist()
    colnames_train = listFea_train[10:]

    # load test data
    df_test = pd.read_csv(test_file)
    listFea_test = df_test.columns.tolist()
    new_listFea_test = listFea_test[:10] + colnames_train
    df_test = df_test.reindex(columns=new_listFea_test)

    # init a BayesianRidge Object
    estimator = BayesianRidge()

    # init a IterativeImputer
    imputer = IterativeImputer(estimator=estimator, random_state=1, max_iter=50)

    # Use bayesian PCA impute missing value
    train_filled = imputer.fit_transform(df_train.iloc[:, 10:])
    test_filled = imputer.transform(df_test.iloc[:, 10:])

    # Turn to dataframe
    train_filled_df = pd.DataFrame(train_filled, columns=colnames_train)
    print("The amount of feature-train(aside from meta information)", train_filled_df.shape[1])
    train_filled_df = pd.concat([df_train.iloc[:, :10], train_filled_df], axis=1)

    test_filled_df = pd.DataFrame(test_filled, columns=colnames_train)
    print("The amount of feature-test(aside from meta information)", test_filled_df.shape[1])
    test_filled_df = pd.concat([df_test.iloc[:, :10], test_filled_df], axis=1)
    
    # save data
    train_filled_df.to_csv(out_train, index=False)
    test_filled_df.to_csv(out_test, index=False)

    # Final log
    total_time = time.time() - start_time
    print(
        f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Completed in {total_time:.2f}s \n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="The input path of training dataset")
    parser.add_argument("--test_file", type=str, help="The input path of test dataset")
    parser.add_argument("--outfile_train", type=str, help="The save path of imputed training dataset")
    parser.add_argument("--outfile_test", type=str, help="The save path of imputed test dataset")
    args = parser.parse_args()
    processMissing(args.train_file, args.test_file, args.outfile_train, args.outfile_test)

"""
# A run-through command:
python 3_processMissingValue.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected.csv \
--test_file ../step1-result/test816_DNA_RNA_Protein_selected.csv \
--outfile_train ../step1-result/train8502_DNA_RNA_Protein_selected_imputed.csv \
--outfile_test ../step1-result/test816_DNA_RNA_Protein_selected_imputed.csv
"""



