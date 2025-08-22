###### IMPORTS #####
import os
import copy
import json
import pdb
import warnings

import numpy as np
from scipy.signal import resample
from scipy.stats import zscore
import mat73
import librosa

import naplib as nl

###### END IMPORTS #####

####### CONFIGURATION & PATHS #######
DATA_PATH = "data"
NEURAL_PATH = os.path.join(DATA_PATH, "neural_data.mat")
LABELS_FILE_PATH = os.path.join(DATA_PATH, "alignment_unattended.json")

SAVE_EMBEDDING_PATH_BASE = "AlignedData/AlignedNeuralData"
SAVE_EMBEDDING_PATH_EMBED = "AlignedData/AlignedEmbeddings"

# Whether to normalize neural signals
NORMALIZE = True
ATTENDED_AND_UNATTENDED_NORMALIZED = True

# Trials to exclude from analysis
EXCLUDE_TRIALS = ['Trial_19', 'Trial_22', 'Trial_26']

# Neural lag configuration
NUMBER_OF_LAGS = 7
LAG_STEP = 5          # in samples
WORD_WINDOW = 20      # number of samples to average after word onset
WORD_OFFSET = 350     # offset for unattended words
#####################################

###############################
# LOAD NEURAL DATA
###############################
neural_data_all_trials = nl.io.import_data(NEURAL_PATH)
neural_data_all_trials_updated = neural_data_all_trials

if ATTENDED_AND_UNATTENDED_NORMALIZED:
    # Load MATLAB structure
    mat = mat73.loadmat(NEURAL_PATH)

    # Generate spectrograms for each trial (here only 1 for example)
    for trial in range(1):
        if trial in [18, 21, 25]:  # Skip corrupted trials
            continue

        neural = mat['out']['resp'][trial][:, :]
        soundf = mat['out']['soundf'][trial]

        # Load attended audio
        clean_att_spec = librosa.load(os.path.join(DATA_PATH, f"Trial_{trial+1}_Conv_1.wav"), sr=soundf)[0]
        clean_att_spec = np.concatenate((np.zeros(int(soundf*0.5)), clean_att_spec, np.zeros(int(soundf*0.5))), axis=0)

        # Load unattended audio
        clean_unatt_spec = librosa.load(os.path.join(DATA_PATH, f"Trial_{trial+1}_Conv_2.wav"), sr=soundf)[0]
        clean_unatt_spec = np.concatenate((np.zeros(int(soundf*3.5)), clean_unatt_spec, np.zeros(int(soundf*0.5))), axis=0)

        # Trim to the shortest duration
        stim_length = min(clean_att_spec.shape[0], clean_unatt_spec.shape[0], mat['out']['sound'][trial].shape[0])
        clean_att_spec = clean_att_spec[:stim_length]
        clean_unatt_spec = clean_unatt_spec[:stim_length]
        neural = neural[:, :round(stim_length/160)]

        neural_data_all_trials_updated[trial]['resp'] = np.transpose(neural)

###########################
# NORMALIZE NEURAL SIGNALS
###########################
if NORMALIZE:
    # Remove trials with NaNs
    nanfree_trial_ids = [np.sum(np.isnan(trial['resp'])) == 0 for trial in neural_data_all_trials]
    neural_data_all_trials_updated = [trial for trial, valid in zip(neural_data_all_trials, nanfree_trial_ids) if valid]

    # Concatenate all trials for normalization
    resp = np.concatenate([d['resp'] for d in neural_data_all_trials_updated], axis=0)

    # Z-score each electrode
    for idx_e in range(resp.shape[1]):
        resp[:, idx_e] = zscore(resp[:, idx_e])

    # Deconcatenate and assign normalized data back to each trial
    start_index = 0
    for trial_data in neural_data_all_trials_updated:
        size = trial_data['resp'].shape[0]
        trial_data['resp'] = resp[start_index:start_index+size, :]
        start_index += size

########################################
# LOAD TRIAL LABELS
########################################
with open(LABELS_FILE_PATH) as f:
    trial_labels = json.load(f)

# Filter excluded trials and reindex
trial_labels = {k: v for k, v in trial_labels.items() if k not in EXCLUDE_TRIALS}
trial_labels = {f'Trial_{i+1}': v for i, (k, v) in enumerate(trial_labels.items())}



###########################
# PROCESS CONTEXTS & TRIALS
###########################
CONTEXT_LENGTHS = ["full"]

for context_len in CONTEXT_LENGTHS:
    # Create folder to save aligned neural data
    save_embedding_path = os.path.join(SAVE_EMBEDDING_PATH_BASE)
    os.makedirs(save_embedding_path, exist_ok=True)

    for idx_trial in range(1):
        # Load embedding for the trial
        try:
            embedding_trial = np.load(
                f"embeddings/Context_{context_len}/Trial_{idx_trial+1}_unattended_embedding.npy",
                allow_pickle=True
            )
        except FileNotFoundError:
            print(f"Alert: No embedding found for trial {idx_trial+1}")
            pdb.set_trace()

        # Initialize word onset lists
        word_onset_list_trial = []
        word_list_trial = []
        absolute_word_count_to_remove = 0
        word_idx_to_remove = []

        # Iterate over words in trial
        for idx_word, word in enumerate(trial_labels[f'Trial_{idx_trial+1}']['words']):
            # Skip words not aligned or starting too early
            if 'start' not in word:
                word_idx_to_remove.append(absolute_word_count_to_remove)
                absolute_word_count_to_remove += 1
                continue
            absolute_word_count_to_remove += 1
            word_onset_list_trial.append(word['start'])
            word_list_trial.append(word['word'])

        # Remove unaligned words from embedding
        embedding_trial = tuple(np.delete(embedding_trial[()][i], word_idx_to_remove, 0) for i in range(len(embedding_trial[()])))

        # Convert word onset times to sample indices
        word_onset_list_trial = [int(round(ele*100)) + WORD_OFFSET for ele in word_onset_list_trial]

        # Copy neural data for processing
        neural_data = copy.deepcopy(neural_data_all_trials_updated[idx_trial])
        neural_data['resp'] = neural_data['resp'][:, :]
        neural_data['subjectID'] = neural_data['subjectID'][:]

        # Process each subject separately
        for subjectID in np.unique(neural_data['subjectID']):
            neural_data_subject = neural_data['resp'][:, neural_data['subjectID']==subjectID]

            # Initialize lagged neural windows
            windows = np.zeros((len(embedding_trial[0]), neural_data_subject.shape[1], NUMBER_OF_LAGS))

            for lag in range(NUMBER_OF_LAGS):
                for idx_word in range(len(embedding_trial[0])):
                    start = word_onset_list_trial[idx_word] + lag*LAG_STEP
                    end = start + WORD_WINDOW
                    if end >= neural_data_subject.shape[0]:
                        continue
                    windows[idx_word, :, lag] = np.mean(neural_data_subject[start:end, :], axis=0)

            # Save embeddings per layer
            for idx_layer, layer_embedding in enumerate(embedding_trial):
                np.save(
                    os.path.join(SAVE_EMBEDDING_PATH_EMBED, f"Trial_{idx_trial+1}_unattended_layer_{idx_layer}_embedding_subject_{subjectID}.npy"),
                    layer_embedding
                )

            # Save aligned neural response per electrode
            for idx_electrode in range(windows.shape[1]):
                np.save(
                    os.path.join(save_embedding_path, f"Trial_{idx_trial+1}_unattended_response_subject_{subjectID}_electrode_{idx_electrode}.npy"),
                    windows[:, idx_electrode, :]
                )

####### END MAIN SCRIPT ######
