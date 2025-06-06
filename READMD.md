# DRP-PSM


> Welcome to DRP-PSM! DRP-PSM is a synonymous variant effect predictors that integrates multi-level features from DNA, RNA, and protein.


## 1 install🔧

```shell
conda env create -f environment.yml
conda activate drp-psm
```


## 2 reproduce💻

Here we will help to reproduce every result in DRP-PSM.
Before we start, you need to change the root path in every `\*.sh` script and every `code/1_selection.py` and `code/foldValidaton.py`:

```shell
...
RNA_ROOT="/YOUR_ABS_PATH/DRP-PSM/RNA_features" # TODO 🚨 Replace this path with your absolute path
...
```


```python
...
with open("/YOUR_PATH/DRP-PSM/DNA_features/code/params_config.yaml") as f: # TODO 🚨 Replace this path as your absolute path
...
```

1. Download whole repository

The original feature data are quite large, you can download the complete version on [Zenodo online](10.5281/zenodo.15606767) 



2. Run feature selection on DNA, RNA, Protein on different sequence length:

The three shell script helps to automatically run through experiment on every sequence length.

```shell
./00-batch-reproduce-drp-psm-DNA.sh
./00-batch-reproduce-drp-psm-RNA.sh
./00-batch-reproduce-drp-psm-Protein.sh
```

3. Concatenate best selected feature and run main model

```shell
python ./DNA_RNA_Protein/code/1_feature_concatenate.py

python ./DNA_RNA_Protein/code/2_train.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected.csv \
--selectedFea ../step2-result/train8502_DART_selected_DRP.csv \
--openfe_features ../step2-result/train8502_OpenFE_generated_DRP.csv \
--model_file ../step2-result/train8502_DART_selected_DRP.model \
--feaName_file ../step2-result/train8502_OpenFE_feaName_DRP.csv \
--lightgbm_importance ../step2-result/train8502_DART_selected_DRP_featureImportance.csv
```

4. Impute missing value with:

```shell
python ./DNA_RNA_Protein/code/3_processMissingValue.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected.csv \
--test_file ../step1-result/test816_DNA_RNA_Protein_selected.csv \
--outfile_train ../step1-result/train8502_DNA_RNA_Protein_selected_imputed.csv \
--outfile_test ../step1-result/test816_DNA_RNA_Protein_selected_imputed.csv
```

5. Predict result:

```shell
python ./DNA_RNA_Protein/code/4_predict_HDPSM.py \
--train_file ../step1-result/train8502_DNA_RNA_Protein_selected_imputed.csv \
--test_file ../step1-result/test816_DNA_RNA_Protein_selected_imputed.csv \
--openfe_features ../step2-result/train8502_OpenFE_generated_DRP.csv \
--selectedFea ../step2-result/train8502_DART_selected_DRP.csv \
--model_file ../step2-result/train8502_DART_selected_DRP.model \
--outfile ../step2-result/test816_predicted_imputed.csv \
--outfile_metrics ../step2-result/test816_predicted_metrics.csv
```

6. Further ablation experiments:

Ablation study on RNA structure feature:

```shell
python ./RNA_features/code/3_train_solely_on_structure.py \
--train_file ../RNA_structure/train8502_RNA_141bp_structurefea.csv \
--selectedFea ../result_structure/reproduce_train8502_141bp_DART_selected_features.csv \
--lightgbm_importance ../result_structure/reproduce_train8502_141bp_selected_importance.csv \
--model_file ../result_structure/reproduce_train8502_141bp_selected.model \
--sequence_len 141bp
```

Ablation study on Protein structure feature:

```shell
python ./Protein_features/code/5_train_solely_on_structure.py \
--train_file ../protein_structure/train8502_17aa_proteinfea.csv \
--selectedFea ../result_structure/reproduce_train8502_protein_17aa_DART_selected_features.csv \
--lightgbm_importance ../result_structure/reproduce_train8502_17aa_selected_importance.csv \
--model_file ../result_structure/reproduce_train8502_17aa_selected.model \
--sequence_len 17aa

```

7. Reproduce every figure:

Please refer to the `./code/1-reproduce-paper-chart/Figure*.ipynb` directory to reproduce every figure in our work.


## 3 citaion📃

If you find our work helpful, please cite:

```
Not published yet.
```