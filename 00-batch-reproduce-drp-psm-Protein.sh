#!/bin/bash

# ====================================
# @description: Batch reproduce result in HDPSM - Protein
# ====================================

# Protein Sequence Length
seq_lengths=(7 17 27 37 47 57 67)
PROTEIN_ROOT="/YOUR_ABS_PATH/DRP-PSM/Protein_features"  # TODO 🚨 Replace this path with your absolute path

# Iterate Protein Sequence Length
for seq in "${seq_lengths[@]}"; do
    echo "[INFO-AA-Feature-Selection] Selecting features with DART and searching best params for ${seq}aa..."

    python ${PROTEIN_ROOT}/code/1_selection.py \
        --train_file ${PROTEIN_ROOT}/protein_seq_structure/train8502_protein_${seq}aa.csv \
        --selectedFea ${PROTEIN_ROOT}/selected_fea/reproduce_train8502_proteinfea_${seq}aa_DART_selected.csv \
        --lightgbm_importance ${PROTEIN_ROOT}/result/reproduce_train8502_proteinfea_${seq}aa_importance.csv \
        --model_file ${PROTEIN_ROOT}/result/reproduce_train8502_proteinfea_${seq}aa.model \
        --sequence_len ${seq}aa

    echo "[INFO-AA-5Fold-Validation] Running 5-fold validation on selected features for ${seq}aa..."

    python ${PROTEIN_ROOT}/code/4_fold_validation_protein.py \
        --train_file ${PROTEIN_ROOT}/protein_seq_structure/train8502_protein_${seq}aa.csv \
        --importance_file ${PROTEIN_ROOT}/result/reproduce_train8502_proteinfea_${seq}aa_importance.csv \
        --png_file ${PROTEIN_ROOT}/5foldValidation_reproduce/train8502_proteinfea_${seq}aa_5fold.png \
        --sequence_len ${seq}aa \
        --prefix ${PROTEIN_ROOT}/5foldValidation_reproduce/train8502_proteinfea_${seq}aa_DART_selected_features_importance_

    echo "----------------------------------------------"
done
echo "[INFO] All sequence lengths processed successfully."
