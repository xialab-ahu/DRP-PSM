#!/bin/bash

# ====================================
# @description: Batch reproduce result in HDPSM - RNA
# ====================================

# RNA Sequence Length
seq_lengths=(21 51 81 111 141 171 201)
RNA_ROOT="/YOUR_ABS_PATH/DRP-PSM/RNA_features" # TODO 🚨 Replace this path with your absolute path

# Iterate RNA Sequence Length
for seq in "${seq_lengths[@]}"; do
    echo "[INFO-RNA-Feature-Selection] Selecting features with DART and searching best params for ${seq}bp..."

    python ${RNA_ROOT}/code/1_selection.py \
        --train_file ${RNA_ROOT}/RNA_seq_structure/train8502_RNA_${seq}bp_allfea.csv \
        --selectedFea ${RNA_ROOT}/result_seq_structure/reproduce_train8502_RNA_${seq}bp_DART_selected.csv \
        --lightgbm_importance ${RNA_ROOT}/result_seq_structure/reproduce_train8502_RNAfea_${seq}bp_importance.csv \
        --model_file ${RNA_ROOT}/result_seq_structure/reproduce_train8502_RNAfea_${seq}bp.model \
        --sequence_len ${seq}bp

    echo "[INFO-RNA-5Fold-Validation] Running 5-fold validation on selected features for ${seq}bp..."

    python ${RNA_ROOT}/code/5_foldValidation.py \
        --train_file ${RNA_ROOT}/RNA_seq_structure/train8502_RNA_${seq}bp_allfea.csv \
        --importance_file ${RNA_ROOT}/result_seq_structure/reproduce_train8502_RNAfea_${seq}bp_importance.csv \
        --png_file ${RNA_ROOT}/5foldValidation/reproduce_train8502_RNAfea_${seq}bp_5fold.png \
        --sequence_len ${seq}bp \
        --prefix ${RNA_ROOT}/5foldValidation/reproduce_train8502_RNAfea_${seq}bp_DART_selected_features_importance_

    echo "----------------------------------------------"
done
echo "[INFO] All sequence lengths processed successfully."
