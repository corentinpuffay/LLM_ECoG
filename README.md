# LLM-ECoG Pipeline

This repository provides a pipeline to align ECoG neural data with LLM embeddings (Mistral-7B). It includes preprocessing, embedding generation, data formatting, and regression analyses.

## Prerequisites

You need the following files:

- A `.mat` file containing the high-gamma neural response to a speech stimulus.
- JSON files with word-level alignments of your transcripts.
- The `.wav` files of your stimuli.

Make sure to change the paths in the scripts accordingly.

## Pipeline Steps

1. **Generate Word Embeddings per Trial**  
   Run `CreateWordEmbeddingPerTrial.py`.  
   This script:
   - Loads neural data and trials' word-level alignments.
   - Generates embeddings for each trial using **Mistral-7B**.
   - Saves embeddings per trial under the `embeddings/` directory.  

   **Dependencies:** `WordEmbeddingGenerator.py` is used here to generate embeddings and handle token-to-word alignment, including removal of repeated words and context length control.

2. **Align Neural Data with Embeddings**  
   Run `ProcessECoGEmbeddings.py`.  
   This script:
   - Loads neural data and the embeddings from step 1.
   - Normalizes neural data.
   - Aligns neural responses with word embeddings, considering subject-specific electrodes and lags.
   - Saves aligned embeddings under `AlignedData/AlignedEmbeddings/` and aligned neural responses under `AlignedData/AlignedNeuralData/`.

3. **Format Data for Regression**  
   Run `ReformatDataForRegression.py`.  
   This script:
   - Iterates over subjects, layers, and trials.
   - Loads aligned embeddings and neural responses.
   - Structures them into `.npy` files suitable for regression analyses.
   - Saves formatted data under `data_formatted_regression/`.

4. **Ridge Regression**  
   Run `RidgeRegression.py`.  
   This script:
   - Performs leave-one-out cross-validation across trials.
   - Runs ridge regression with bootstrap-based model selection.
   - Iterates over subjects and layers.
   - Saves trial-by-trial correlation results per subject under `results_regression/`.

