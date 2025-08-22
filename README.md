# LLM-ECoG Pipeline

This repository contains scripts to process ECoG data aligned with LLM embeddings and perform ridge regression analyses.

## Requirements

Before running the pipeline, you need:

- A `.mat` file containing high-gamma ECoG responses to a speech stimulus.
- Word-level alignments of your transcripts in JSON format.
- The `.wav` files of your speech stimuli.

> **Note:** Make sure to adjust all paths in the scripts to match your local folder structure.

## Pipeline Steps

Run the following scripts in sequence to process your data:

1. **Create Word Embeddings per Trial**  
   Run `CreateWordEmbeddingPerTrial.py` to generate embeddings for each trial of your experiment based on your transcripts.  

2. **Align Neural Data with LLM Embeddings**  
   Run `ProcessECoGEmbeddings.py` to align neural responses with word-level embeddings. This step maps each word in your transcripts to the corresponding neural response at each electrode and across multiple lags.  

3. **Run Ridge Regression**  
   Use `RidgeRegression.py` to perform ridge regression analyses linking the aligned embeddings to neural responses. This script performs leave-one-out cross-validation across trials and stores correlations per layer, electrode, and subject.

**Command Example for all steps:**
```bash
python CreateWordEmbeddingPerTrial.py
python ProcessECoGEmbeddings.py
python RidgeRegression.py
