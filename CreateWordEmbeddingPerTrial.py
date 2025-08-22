"""
ECoG Spectrogram and Mistral-7B Embedding Pipeline

This script:
1. Loads neural (ECoG) data and trial labels.
2. Generates auditory spectrograms for attended/unattended stimuli.
3. Normalizes neural and spectrogram data.
4. Prepares embeddings using Mistral-7B model for various context lengths.

Author: Corentin Puffay
"""

###### IMPORTS #####
import os
import pdb
import re
import copy
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
import mat73
from scipy.signal import resample

import naplib as nl
import WordEmbeddingGenerator
from WordEmbeddingGenerator import generate_embedding_mistral
from huggingface_hub import login

# Replace "YOUR_TOKEN_HERE" with your actual Hugging Face token
login(token="your_token")

###### CONFIGURATION #####
NEURAL_PATH = "data/neural_data.mat"  # replace with your .mat file path
ALIGNMENT_FILE_PATH = "data/alignment.json"  # replace with your JSON labels file path
SAVE_EMBEDDING_DIR = "embeddings"  # replace with desired save directory

# Flags
NORMALIZE = True
NORMALIZE_PER_TRIAL = False
ATTENDED_AND_UNATTENDED_NORMALIZED = True

# Context lengths for embeddings
#CONTEXT_LENGTHS = [1, 5, 10, 15, 50]
CONTEXT_LENGTHS = ["full"]
FULL_OR_REDUCED = "full"

###### LOAD DATA #####
neural_data_all_trials = nl.io.import_data(NEURAL_PATH)

###### TRIMMING NEURAL DATA AND STIMULI TO SHORTEST STIMULUS #####
if ATTENDED_AND_UNATTENDED_NORMALIZED:
    neural_data_all_trials_updated = neural_data_all_trials
    mat_data = mat73.loadmat(NEURAL_PATH)


    for trial_idx in range(28):
        if trial_idx in [18, 21, 25]:  # Exclude corrupted trials
            continue

        # Neural and stimulus data
        neural = mat_data['out']['resp'][trial_idx][:, :]
        soundf = mat_data['out']['soundf'][trial_idx]

        # Load attended/unattended audio
        att_path = f"data/Trial_{trial_idx + 1}_Conv_1.wav"
        unatt_path = f"data/Trial_{trial_idx + 1}_Conv_2.wav"

        clean_att = librosa.load(att_path, sr=soundf)[0]
        clean_att = np.concatenate((np.zeros(int(soundf*0.5)), clean_att, np.zeros(int(soundf*0.5))), axis=0)
        clean_unatt = librosa.load(unatt_path, sr=soundf)[0]
        clean_unatt = np.concatenate((np.zeros(int(soundf*3.5)), clean_unatt, np.zeros(int(soundf*0.5))), axis=0)

        clean_unatt = librosa.load(unatt_path, sr=soundf)[0]
        clean_unatt = np.concatenate((np.zeros(int(soundf*3.5)), clean_unatt, np.zeros(int(soundf*0.5))), axis=0)

        # Trim to shortest length
        stim_length = min(clean_att.shape[0], clean_unatt.shape[0], mat_data['out']['sound'][trial_idx].shape[0])
        clean_att = clean_att[:stim_length]
        clean_unatt = clean_unatt[:stim_length]
        neural = neural[:, :round(stim_length / 160)]

        # Store
        neural_data_all_trials_updated[trial_idx]['resp'] = np.transpose(neural)


###### NORMALIZATION #####
if NORMALIZE:
    nanfree_ids = [np.sum(np.isnan(trial['resp'])) == 0 for trial in neural_data_all_trials_updated]
    neural_data_all_trials_updated = [trial for trial, include in zip(neural_data_all_trials_updated, nanfree_ids) if include]

    # Reduce spectrogram dimension
    resample_kwargs = {'num': 32, 'axis': 1}
    neural_data_all_trials_updated = nl.Data(neural_data_all_trials_updated)
    #neural_data_all_trials_updated['spec_32_att'] = nl.array_ops.concat_apply(neural_data_all_trials_updated['spec_att'], resample, function_kwargs=resample_kwargs)
    #neural_data_all_trials_updated['spec_32_unatt'] = nl.array_ops.concat_apply(neural_data_all_trials_updated['spec_unatt'], resample, function_kwargs=resample_kwargs)

    # Concatenate for z-scoring
    resp = copy.deepcopy(np.concatenate([d['resp'] for d in neural_data_all_trials_updated], axis=0))
    #stim_att = copy.deepcopy(np.concatenate([d['spec_32_att'] for d in neural_data_all_trials_updated], axis=0))
    #stim_unatt = copy.deepcopy(np.concatenate([d['spec_32_unatt'] for d in neural_data_all_trials_updated], axis=0))

    # Z-score
    for idx in range(resp.shape[1]):
        resp[:, idx] = (resp[:, idx] - np.mean(resp[:, idx])) / np.std(resp[:, idx])
    #stim_att = nl.preprocessing.normalize([stim_att], axis=1)[0]
    #stim_unatt = nl.preprocessing.normalize([stim_unatt], axis=1)[0]

    # Deconcatenate per trial
    start_idx = 0
    for idx, trial in enumerate(neural_data_all_trials_updated):
        size = trial['resp'].shape[0]
        trial['resp'] = resp[start_idx:start_idx+size, :]
        #trial['spec_32_att'] = stim_att[start_idx:start_idx+size, :]
        #trial['spec_32_unatt'] = stim_unatt[start_idx:start_idx+size, :]
        start_idx += size

###### LOAD TRIAL LABELS #####
with open(ALIGNMENT_FILE_PATH) as f:
    trial_labels = json.load(f)

###### BUILD TRIAL DICTIONARY #####
trial_dict = {}
for i, trial in enumerate(neural_data_all_trials_updated):
    data = np.sum(trial['speakerMatrix'], axis=1)
    attended = [int(j) for j, val in enumerate(data) if val > 0]
    unattended = [int(j) for j, val in enumerate(data) if val < 0]
    trial_dict[f'Trial {i}'] = {'attended': attended, 'unattended': unattended}

with open('trial_dict.json', 'w') as json_file:
    json.dump(trial_dict, json_file, indent=4)

###### GENERATE EMBEDDINGS #####
for context_len in CONTEXT_LENGTHS:
    save_path = os.path.join(SAVE_EMBEDDING_DIR, f"Context_{context_len}")
    os.makedirs(save_path, exist_ok=True)

    idx_length_tab = []
    trial_length_tab = []

    for trial_idx in range(28):
        if trial_idx in [18, 21, 25]:
            continue

        transcript_trial = trial_labels[f"Trial_{trial_idx+1}"]['transcript']
        transcript_trial = re.sub(r'\b(\w+)-(\w+)\b', r'\1 \2', transcript_trial)

        print(f"Processing Trial {trial_idx+1}: {transcript_trial}")

        embedding_trial = generate_embedding_mistral(transcript_trial, [], context_len, FULL_OR_REDUCED, trial_idx)

        # Save word embeddings for each trial
        np.save(os.path.join(SAVE_EMBEDDING_DIR, f"Trial_{trial_idx+1}_embedding.npy"), embedding_trial)



