# CompressedEmbeddings and Classifier

This module provides `CompressedEmbeddings` and `CompressedEmbeddingsClassifier` classes that use ESM2 from the Transformers library with compression from the cheap-proteins algorithm.

## Features

1. **CompressedEmbeddings**: Encoder that takes amino acid sequences and produces compressed embeddings using:
   - ESM2 from Transformers library for initial embeddings
   - cheap-proteins compression algorithm for dimensionality reduction

2. **CompressedEmbeddingsClassifier**: Binary classifier for pairs of sequences that:
   - Uses CompressedEmbeddings as encoder (shared)
   - Generates compressed embeddings for two sequences separately
   - Averages each embedding along sequence length dimension
   - Concatenates the two averaged embeddings
   - Passes through MLP with softmax for binary prediction

## Available Compression Levels

The compression level is specified by two parameters:

- **shorten_factor**: Either `1` or `2`
- **dimension**: One of `4, 8, 16, 32, 64, 128, 256, 512, 1024`

**Note**: Not all combinations are available. The available combinations are:

### shorten_factor = 1:
- Dimensions: 4, 8, 16, 32, 64, 128, 256, 512

### shorten_factor = 2:
- Dimensions: 4, 8, 16, 32, 64, 128, 256, 512, 1024

## Installation

Make sure you have the required dependencies:

```bash
pip install torch transformers
```

The cheap-proteins code should be in the `cheap-proteins/` directory relative to this file.

## Usage

### Basic Usage: CompressedEmbeddings

```python
from compressed_embeddings import CompressedEmbeddings

# Initialize encoder
# Note: Only ESM2 3B model is supported
encoder = CompressedEmbeddings(
    shorten_factor=1,
    dimension=64,
    esm2_model_name="facebook/esm2_t36_3B_UR50D",  # Must be ESM2 3B
    device="cuda"  # or "cpu"
)

# Encode sequences
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
    "VFGRCELAAAMRHGLDNYRGYSLGNWVCAAFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKIVSDGNGMNAWVAWRNRCGTDVQAWIRGCRL",
]

compressed_emb, mask = encoder.encode(sequences)
print(f"Compressed embeddings shape: {compressed_emb.shape}")  # (batch, compressed_seq_len, dimension)
print(f"Mask shape: {mask.shape}")  # (batch, compressed_seq_len)
```

### Using the Classifier

The classifier takes **two sequences** as input and predicts binary classification based on their combined embeddings.

```python
from compressed_embeddings import CompressedEmbeddingsClassifier

# Initialize classifier
# Note: Only ESM2 3B model is supported
classifier = CompressedEmbeddingsClassifier(
    shorten_factor=1,
    dimension=64,
    esm2_model_name="facebook/esm2_t36_3B_UR50D",  # Must be ESM2 3B
    mlp_hidden_dims=[256, 128],  # Hidden layer dimensions
    dropout=0.1,
    device="cuda"
)

# Prepare pairs of sequences
sequence1_list = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV"]
sequence2_list = ["RTDCYGNVNRIDTTGASCKTAKPEGLSYCGVSASKKIAERDLQAMDRYKTIIKKVGEKLCVEPAVIAGIISRESHAGKVLKNGWGDRGNGFGLMQVDKRSHKPQGTWNGEVHITQGTTILINFIKTIQKKFPSWTKDQQLKGGISAYNAGAGNVRSYARMDIGTTHDDYANDVVARAQYYKQHGY"]

# Get predictions for pairs of sequences
logits = classifier(sequence1_list, sequence2_list)  # Shape: (batch, 2)

# Get probabilities
probs = classifier.predict_proba(sequence1_list, sequence2_list)  # Shape: (batch, 2)

# Get class predictions
preds = classifier.predict(sequence1_list, sequence2_list)  # Shape: (batch,)

# Get embeddings along with predictions
logits_with_emb, emb1, emb2 = classifier(sequence1_list, sequence2_list, return_embeddings=True)
# emb1 and emb2 are of shape (batch, dimension)
```

## Architecture Details

### CompressedEmbeddings

1. **Tokenization**: Uses ESM2 tokenizer from Transformers to tokenize amino acid sequences
2. **ESM2 Embedding**: Extracts embeddings from all layers of ESM2 model
3. **Layer Combination**: Combines all layer outputs using learned weights (similar to ESMFold)
4. **Projection**: Projects to 1024 dimensions (matching ESMFold's processing)
5. **Normalization**: Applies latent scaling (channel-wise min-max normalization)
6. **Compression**: Applies cheap-proteins hourglass compression model

### CompressedEmbeddingsClassifier

1. **Encoder**: Uses CompressedEmbeddings to get compressed embeddings for two sequences separately (L×D for each)
2. **Averaging**: Averages each embedding along sequence length dimension to get D-dimensional vectors
3. **Concatenation**: Concatenates the two averaged embeddings (2×D)
4. **MLP**: Multi-layer perceptron with configurable hidden dimensions (input: 2×D)
5. **Output**: Binary classification with softmax (2 classes)

## Model Parameters

- **shorten_factor**: Compression factor (1 or 2). Higher values compress more aggressively.
- **dimension**: Final embedding dimension after compression
- **esm2_model_name**: ESM2 model from HuggingFace. **Must be "facebook/esm2_t36_3B_UR50D"** to match ESMFold's 3B model configuration. Other models are not supported.
- **mlp_hidden_dims**: List of hidden layer dimensions for the classifier MLP
- **dropout**: Dropout rate for the classifier MLP

## Important Note

**Only ESM2 3B model is supported.** The code will raise a `ValueError` if any other ESM2 model is specified. This is because:
- ESMFold pretrained weights are specifically trained with ESM2 3B (36 layers, 2560 embedding dimension)
- The projection layers (`esm_s_combine` and `esm_s_mlp`) are loaded from ESMFold and require exact model configuration matching
- Using a different model would result in dimension mismatches and incorrect embeddings

## Notes

- The compression models are automatically downloaded from HuggingFace on first use
- Models are cached in `~/.cache/cheap` (or `CHEAP_CACHE` environment variable)
- ESM2 models are downloaded from HuggingFace automatically
- The encoder processes sequences in batches and automatically pads to the longest sequence
- Masks indicate valid (non-padded) positions in the compressed embeddings

## Example

See `example_usage.py` for a complete working example.

