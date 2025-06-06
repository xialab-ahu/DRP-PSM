"""
@description: This script concatenates the best DNA-RNA-Protein features together
From the aforementioned results, best DNA-fea is 111bp, best RNA-fea is 141bp, best Protein-fea is 17aa
"""

import pandas as pd


def concatenate(dna_fea_df, rna_fea_df, protein_fea_df):
    """
    Desc:
        Concatenate DNA RNA Protein features
    Args:
        dna_fea_df
        rna_fea_df
        protein_fea_df
    """
    df = pd.concat([dna_fea_df, rna_fea_df], axis=1)
    df = pd.concat([df, protein_fea_df], axis=1)
    return df

def filter_data(all_fea_path, dart_selected_path):
    """
    Desc:
        filter dart-selected feature
    Args:
        all_fea, the original all feature file
        dart_selected, the dart_selected feature name
    """
    df = pd.read_csv(all_fea_path, low_memory=False)
    selected_feature = pd.read_csv(dart_selected_path)['feature'].tolist()
    df_meta = df.iloc[:, :10]
    df_feature = df[selected_feature]
    return df_feature, df_meta
    

# 1 DNA feature 111bp -> 13dim
# Train 
dna_train_path = "../../DNA_features/allfea/train8502_DNAfea_111bp.csv"
dna_dart_selected_path = "../../DNA_features/selected_fea/reproduce_train8502_DNAfea_111bp_DART_selected.csv"
# Test
dna_test_path = "../../DNA_features/allfea/test816_DNAfea_111bp.csv"
# Selected
dna_train_selected, dna_train_meta = filter_data(dna_train_path, dna_dart_selected_path)
dna_test_selected, dna_test_meta = filter_data(dna_test_path, dna_dart_selected_path)
print(f"[INFO] DNA feature dim: {dna_test_selected.shape}")

# 2 RNA feature 141bp -> 27dim
# Train
rna_train_path = "../../RNA_features/RNA_seq_structure/train8502_RNA_141bp_allfea.csv"
rna_dart_selected_path = "../../RNA_features/result_seq_structure/reproduce_train8502_RNA_141bp_DART_selected.csv"
# Test
rna_test_path = "../../RNA_features/RNA_seq_structure/test816_RNA_141bp_allfea.csv"
# Selected
rna_train_selected, _ = filter_data(rna_train_path, rna_dart_selected_path)
rna_test_selected, _ = filter_data(rna_test_path, rna_dart_selected_path)
print(f"[INFO] RNA feature dim: {rna_test_selected.shape}")

# 3 Protein feature 17aa -> 24dim
# Train
protein_train_path = "../../Protein_features/protein_seq_structure/train8502_protein_17aa.csv"
protein_dart_selected_path = "../../Protein_features/selected_fea/reproduce_train8502_proteinfea_17aa_DART_selected.csv"
# Test
protein_test_path = "../../Protein_features/protein_seq_structure/test816_protein_17aa.csv"
# Selected
protein_train_selected, _ = filter_data(protein_train_path, protein_dart_selected_path)
protein_test_selected, _ = filter_data(protein_test_path, protein_dart_selected_path)
print(f"[INFO] Protein feature dim: {protein_test_selected.shape}")

# 4 Concatenate 
final_train_selected = concatenate(dna_train_selected, rna_train_selected, protein_train_selected)
final_test_selected = concatenate(dna_test_selected, rna_test_selected, protein_test_selected)

final_train_selected = pd.concat([dna_train_meta, final_train_selected], axis=1)
final_test_selected = pd.concat([dna_test_meta, final_test_selected], axis=1)
final_train_selected.to_csv("../step1-result/train8502_DNA_RNA_Protein_selected.csv", index=False)
final_test_selected.to_csv("../step1-result/test816_DNA_RNA_Protein_selected.csv", index=False)

