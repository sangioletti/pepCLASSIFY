"""
Example usage of CompressedEmbeddings and CompressedEmbeddingsClassifier
"""

import torch
from compressed_embeddings import CompressedEmbeddings, CompressedEmbeddingsClassifier

# Example 1: Using CompressedEmbeddings directly
print("=" * 60)
print("Example 1: CompressedEmbeddings")
print("=" * 60)

# Initialize encoder with compression level
# shorten_factor: 1 or 2
# dimension: 4, 8, 16, 32, 64, 128, 256, 512, 1024
# Note: Only ESM2 3B model is supported
encoder = CompressedEmbeddings(
    shorten_factor=1,
    dimension=64,
    esm2_model_name="facebook/esm2_t36_3B_UR50D",  # Must be ESM2 3B
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Encode sequences
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
    "VFGRCELAAAMRHGLDNYRGYSLGNWVCAAFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKIVSDGNGMNAWVAWRNRCGTDVQAWIRGCRL",
]

compressed_emb, mask = encoder.encode(sequences)
print(f"Compressed embeddings shape: {compressed_emb.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")

# Example 2: Using CompressedEmbeddingsClassifier
print("\n" + "=" * 60)
print("Example 2: CompressedEmbeddingsClassifier")
print("=" * 60)

# Initialize classifier
# Note: Only ESM2 3B model is supported
classifier = CompressedEmbeddingsClassifier(
    shorten_factor=1,
    dimension=64,
    esm2_model_name="facebook/esm2_t36_3B_UR50D",  # Must be ESM2 3B
    mlp_hidden_dims=[256, 128],
    dropout=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prepare pairs of sequences for classification
# The classifier takes two sequences and predicts if they belong to the same class
sequence1_list = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
    "VFGRCELAAAMRHGLDNYRGYSLGNWVCAAFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKIVSDGNGMNAWVAWRNRCGTDVQAWIRGCRL",
]

sequence2_list = [
    "RTDCYGNVNRIDTTGASCKTAKPEGLSYCGVSASKKIAERDLQAMDRYKTIIKKVGEKLCVEPAVIAGIISRESHAGKVLKNGWGDRGNGFGLMQVDKRSHKPQGTWNGEVHITQGTTILINFIKTIQKKFPSWTKDQQLKGGISAYNAGAGNVRSYARMDIGTTHDDYANDVVARAQYYKQHGY",
    "AYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
]

# Get predictions for pairs of sequences
logits = classifier(sequence1_list, sequence2_list)
print(f"Logits shape: {logits.shape}")

# Get probabilities
probs = classifier.predict_proba(sequence1_list, sequence2_list)
print(f"Probabilities shape: {probs.shape}")
print(f"Probabilities:\n{probs}")

# Get class predictions
preds = classifier.predict(sequence1_list, sequence2_list)
print(f"Predictions: {preds}")

# Get embeddings along with predictions
logits_with_emb, emb1, emb2 = classifier(sequence1_list, sequence2_list, return_embeddings=True)
print(f"Embeddings 1 shape: {emb1.shape}")
print(f"Embeddings 2 shape: {emb2.shape}")

print("\n" + "=" * 60)
print("Example 3: Single pair of sequences")
print("=" * 60)

# Single pair of sequences
single_seq1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV"
single_seq2 = "RTDCYGNVNRIDTTGASCKTAKPEGLSYCGVSASKKIAERDLQAMDRYKTIIKKVGEKLCVEPAVIAGIISRESHAGKVLKNGWGDRGNGFGLMQVDKRSHKPQGTWNGEVHITQGTTILINFIKTIQKKFPSWTKDQQLKGGISAYNAGAGNVRSYARMDIGTTHDDYANDVVARAQYYKQHGY"

compressed_emb_single1, mask_single1 = encoder.encode(single_seq1)
compressed_emb_single2, mask_single2 = encoder.encode(single_seq2)
print(f"Single sequence 1 compressed embeddings shape: {compressed_emb_single1.shape}")
print(f"Single sequence 2 compressed embeddings shape: {compressed_emb_single2.shape}")

probs_single = classifier.predict_proba(single_seq1, single_seq2)
print(f"Single pair probabilities: {probs_single}")

