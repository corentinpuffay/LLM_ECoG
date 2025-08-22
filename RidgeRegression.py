#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge regression leave-one-out cross-validation for neural data alignment.

This script:
1. Iterates over context lengths, subjects, and layers.
2. Performs leave-one-out cross-validation across trials.
3. Runs ridge regression with bootstrap-based model selection.
4. Saves raw trial-by-trial correlation results per subject.

Dependencies:
- numpy
- matplotlib
- scipy
- ridge_utils (custom package with ridge + bootstrap_ridge)
"""

import os
import numpy as np
from ridge_utils.ridge import bootstrap_ridge
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
CONTEXT_LENGTHS = ["full"]
SAVE_BASE = "results_regression"
DATA_BASE = "data_formatted_regression"

N_SUBJECTS = 3
N_LAYERS = 33
N_TRIALS = 25

# Ridge regression parameters
ALPHAS = np.logspace(-2, 4, 8)   # Regularization values (10^-2 ... 10^4)
NBOOTS = 10                      # Bootstrap resamples
CHUNKLEN = 100                   # Chunk length for bootstrap


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
print("Python script started...", flush=True)

for context_len in CONTEXT_LENGTHS:

    save_dir = os.path.join(SAVE_BASE)
    if not os.path.exists(save_dir):
        print(f"ERROR: Saving path does not exist -> {save_dir}")
        break

    # Iterate over subjects
    for subject in range(1, N_SUBJECTS + 1):
        all_corrs_all_layers = []

        # Iterate over layers
        for layer_number in range(N_LAYERS):
            all_corrs = []

            # Leave-one-out cross-validation across trials
            for leave_out in range(N_TRIALS):
                acc_stim, acc_resp = [], []

                # Build training data (exclude one trial)
                for i in range(N_TRIALS):
                    if i == leave_out:
                        continue

                    stim_path = os.path.join(
                        DATA_BASE,
                        f"Context_{context_len}_trimmed_4.0_unattended/Subject_{subject}/Layer_{layer_number}/Trial_{i}_embedding.npy"
                    )
                    resp_path = os.path.join(
                        DATA_BASE,
                        f"Context_{context_len}_trimmed_4.0_unattended/Subject_{subject}/Trial_{i}_response.npy"
                    )

                    acc_stim.append(np.load(stim_path))
                    arr = np.load(resp_path)
                    acc_resp.append(arr.reshape(arr.shape[0], -1))

                # Stack training data
                Rstim = np.vstack(acc_stim)
                Rresp = np.vstack(acc_resp)

                # Load held-out trial
                Pstim = np.load(
                    os.path.join(
                        DATA_BASE,
                        f"Context_{context_len}_trimmed_4.0_unattended/Subject_{subject}/Layer_{layer_number}/Trial_{leave_out}_embedding.npy"
                    )
                )
                arr = np.load(
                    os.path.join(
                        DATA_BASE,
                        f"Context_{context_len}_trimmed_4.0_unattended/Subject_{subject}/Layer_{layer_number}/Trial_{leave_out}_response.npy"
                    )
                )
                Presp = arr.reshape(arr.shape[0], -1)
                n_electrodes = arr.shape[1]

                # Define number of bootstrap chunks
                nchunks = int(len(Rresp) * 0.25 / CHUNKLEN)

                # Debug shapes
                print(Rstim.shape, Pstim.shape, Rresp.shape, Presp.shape, flush=True)

                # Run ridge regression with bootstrap-based model selection
                _, corr, _, _, _ = bootstrap_ridge(
                    Rstim, Rresp, Pstim, Presp,
                    alphas=ALPHAS, nboots=NBOOTS,
                    chunklen=CHUNKLEN, nchunks=nchunks,
                    use_corr=True, single_alpha=False
                )

                # Reshape correlations to (electrodes × lags)
                all_corrs.append(np.reshape(corr, (n_electrodes, 7)))

            # Collect results for this layer
            all_corrs_all_layers.append(all_corrs)
            print(f"Finished Layer {layer_number}, Subject {subject}, Context {context_len}", flush=True)

        # Save results for this subject
        save_path = os.path.join(save_dir, f"Subject_{subject}_all_layers.npy")
        np.save(save_path, all_corrs_all_layers)
        print(f"✅ Saved results: {save_path}")


print("Script completed successfully.")
