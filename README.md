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
   - Loads neural data and trials' word-level alignments.  
   - Normalizes neural data.  
   - Prepares embeddings using the Mistral-7B model for a given context length.  
   - Saves word embeddings per trial in `embeddings/`.

2. **Align Neural Data with LLM Embeddings**  
   Run `ProcessECoGEmbeddings.py` to align neural responses with word-level embeddings.  
   - Creates word-level alignment between neural data and LLM embeddings.  
   - Saves aligned embeddings per trial, layer, and subject in `AlignedData/AlignedEmbeddings/`.  
   - Saves aligned neural responses per trial, electrode, and subject in `AlignedData/AlignedNeuralData/`.

3. **Reformat Data for Regression**  
   Run `ReformatDataForRegression.py` to reshape and organize the aligned embeddings and neural responses into regression-ready files.  
   - Saves `.npy` files per trial, per layer, and per subject in `data_formatted_regression/`.

4. **Run Ridge Regression**  
   Run `RidgeRegression.py` to perform ridge regression analyses linking the regression-ready embeddings to neural responses.  
   - Iterates over context lengths, subjects, and layers.  
   - Performs leave-one-trial-out cross-validation.  
   - Runs ridge regression with bootstrap-based model selection (`ridge_utils.bootstrap_ridge`).  
   - Saves raw trial-by-trial correlation results per subject in `results_regression/`.

**Command Example for all steps:**
```bash
python CreateWordEmbeddingPerTrial.py
python ProcessECoGEmbeddings.py
python ReformatDataForRegression.py
python RidgeRegression.py
