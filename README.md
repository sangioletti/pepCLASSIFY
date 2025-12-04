Training a classifier to recognise peptide binders given a sequence, using compressed ESM2 embeddings

We do this in the following way:

1) Encode both the peptide and the protein using esm2 + cheap embedding. We use the cheap to drastically reduce dimensionality (hopefully better to avoid overfitting)

2) The concatenated embeddings are passed to an MLP, ending with a softmax for binary classification

3) We train on the database in PEP_MIMIC_SPR_DATA. Entries with a KD are considered positive (1), the other negatives (0). The boundary for the experiments was KD< 100 nM
