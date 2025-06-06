#!/bin/bash

# ====================================
# @description: Batch reproduce result in DRP-PSM - DNA
# ====================================

# DNA sequence length
seq_lengths=(21 51 81 111 141 171 201)
DNA_ROOT="/YOUR_ABS_PATH/DRP-PSM/DNA_features" # TODO 🚨 Replace this path with your absolute path

# 遍历 DNA 
for seq in "${seq_lengths[@]}"; do
    echo "[INFO-DNA-Feature-Selection] Selecting features with DART and searching best params for ${seq}bp..."

    python ${DNA_ROOT}/code/1_selection_dart.py \
        --train_file ${DNA_ROOT}/allfea/train8502_DNAfea_${seq}bp.csv \
        --selectedFea ${DNA_ROOT}/selected_fea/reproduce_train8502_DNAfea_${seq}bp_DART_selected.csv \
        --lightgbm_importance ${DNA_ROOT}/result/reproduce_train8502_DNAfea_${seq}bp_importance.csv \
        --model_file ${DNA_ROOT}/result/reproduce_train8502_DNAfea_${seq}bp.model \
        --sequence_len ${seq}bp

    echo "[INFO-DNA-5Fold-Validation] Running 5-fold validation on selected features for ${seq}bp..."

    python ${DNA_ROOT}/code/3_foldValidation.py \
        --train_file ${DNA_ROOT}/allfea/train8502_DNAfea_${seq}bp.csv \
        --importance_file ${DNA_ROOT}/result/reproduce_train8502_DNAfea_${seq}bp_importance.csv \
        --png_file ${DNA_ROOT}/fig_5foldValidation/reproduce_train8502_DNAfea_${seq}bp_5fold.png \
        --sequence_len ${seq}bp \
        --prefix ${DNA_ROOT}/fig_5foldValidation/train8502_DNAfea_${seq}bp_DART_selected_features_importance_

    echo "----------------------------------------------"
done
echo "[INFO] All sequence lengths processed successfully."
