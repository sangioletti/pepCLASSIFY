# Training the CompressedEmbeddingsClassifier

This guide explains how to train the classifier using the data from `PEP_MIMIC_SPR_DATA.xlsx`.

## Overview

The training process consists of two steps:

1. **Data Preparation**: Extract sequences from Excel file, download FASTA files, and create a training dataset
2. **Training**: Train the classifier on the prepared data

## Step 1: Prepare Training Data

First, run the data preparation script to create the training dataset:

```bash
python prepare_training_data.py
```

This script will:
- Read `PEP_MIMIC_SPR_DATA.xlsx`
- Extract PDB codes from the first column (part before first "_")
- Download FASTA files from PDB for each complex
- Identify the target chain (longer sequence) and binder chain
- Extract peptide sequences from column 2
- Create binary labels based on KD column (1 if KD has a number, 0 otherwise)
- Save results to `training_data.txt` with format: `target_sequence\tpeptide_sequence\tlabel`

The output file `training_data.txt` will have 3 columns:
- `target_sequence`: The longer chain sequence from the PDB complex
- `peptide_sequence`: The peptide sequence from column 2 of the Excel file
- `label`: Binary label (0 or 1) based on whether KD column has a value

**Note**: You can check the `training_data.txt` file before proceeding to training to verify the data is correct.

## Step 2: Train the Classifier

Once the data is prepared, run the training script:

```bash
python train_classifier.py
```

This script will:
- Load the prepared dataset from `training_data.txt`
- Split into 80% training and 20% test sets
- Initialize the CompressedEmbeddingsClassifier
- Train for 10 epochs with Adam optimizer
- Save the trained model to `trained_classifier.pt`

### Training Parameters

You can modify these parameters in `train_classifier.py`:

- `num_epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 4, adjust based on GPU memory)
- `learning_rate`: Learning rate (default: 1e-4)
- `shorten_factor`: Compression shorten factor (default: 1)
- `dimension`: Compression dimension (default: 64)
- `mlp_hidden_dims`: MLP hidden layer dimensions (default: [256, 128])

### Model Architecture

The classifier:
- Uses ESM2 3B model for embeddings (frozen)
- Applies cheap-proteins compression (frozen)
- Trains only the MLP layers for binary classification

## Output Files

- `training_data.txt`: Prepared dataset with sequences and labels
- `fasta_files/`: Directory containing downloaded FASTA files
- `trained_classifier.pt`: Trained model weights

## Requirements

Make sure you have:
- `pandas` and `openpyxl` for reading Excel files
- `torch` for training
- `tqdm` for progress bars
- Internet connection for downloading FASTA files from PDB

## Troubleshooting

1. **Excel file reading errors**: Install/upgrade `openpyxl`: `pip install openpyxl>=3.1.0`

2. **FASTA download failures**: Some PDB codes might not be available. The script will skip these and continue.

3. **Memory issues**: Reduce `batch_size` in the training script if you run out of GPU memory.

4. **Slow training**: The ESM2 3B model is large. Consider using a GPU for faster training.

