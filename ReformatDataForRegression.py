###### IMPORTS #####
import warnings
import numpy as np
import textgrid
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datasets
import transformers
import torch
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pdb
import os
import string
from collections import defaultdict
import numpy as np
from hdf5storage import savemat, loadmat
from scipy.stats import pearsonr, spearmanr, bootstrap
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import cross_val_predict
import pdb
import naplib as nl
from datasets import load_dataset
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score
import scipy
from sklearn.model_selection import ShuffleSplit
from sklearn.manifold import TSNE
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, LeaveOneOut
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
import pandas as pd
import scipy
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, shapiro
import mne
import seaborn as sns
from IPython.display import Markdown, display
from pandas.api.types import is_numeric_dtype
import sympy
from scipy.signal import find_peaks
from IPython.display import display
from sklearn import preprocessing as prep
import sklearn.metrics as skm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import h5py
import hdf5storage
import json
import copy
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, mean_squared_error

###### END IMPORTS #####


####### MAIN SCRIPT ######

# Amount of connected electrodes per subject (after dropping NaN values)
len_df_subjects = [166, 308, 115, 191, 150, 151]
CONTEXT_LENGTHS = ["full"]

# Iterate over context_length
for context_len in CONTEXT_LENGTHS:

    print('CONTEXT = {}'.format(context_len))

    load_embedding_path = "AlignedData/AlignedEmbeddings"
    load_ND_path = "AlignedData/AlignedNeuralData"

    if not os.path.exists("data_formatted_regression"):
        print("CREATE Context_{} folder".format(context_len))
        os.makedirs("data_formatted_regression")

    for layer_number in range(0, 33):
        print(layer_number)

        correlations_all_subjects = []

        # Iterate over subjects
        for subject in range(1, 7):

            if not os.path.exists("data_formatted_regression/Subject_{}/Layer_{}".format(subject, layer_number)):
                os.makedirs("data_formatted_regression/Subject_{}/Layer_{}".format(subject, layer_number))

            #### RESUME HERE FOR EACH ELECTRODE ####
            # Harvest electrodes for subject from Trial 1 files
            electrode_list_subject = []
            for filename in os.listdir(load_ND_path):
                if filename.split('.')[0].split('_')[1] == '1' and filename.split('.')[0].split('_')[-1] == str(
                        int(subject)):
                    electrode_list_subject.append(filename.split('.')[1].split('_')[-1])

            ## Reduce electrodes to the ones from the excel sheet
            print("ELECTRODE LENGTH BEFORE : {}".format(len(electrode_list_subject)))
            electrode_list_subject = electrode_list_subject[0:len_df_subjects[subject - 1]]
            print("ELECTRODE LENGTH AFTER : {}".format(len(electrode_list_subject)))

            embedding_trials = []
            resp_trials_list = []

            for trial_idx in range(0, 1):

                # 1. Embeddings
                embedding_trial = np.load(os.path.join(load_embedding_path,
                                                       "Trial_{}_unattended_layer_{}_embedding_subject_{}.npy".format(
                                                           trial_idx + 1, int(layer_number), float(subject))))
                embedding_trials.append(embedding_trial)

                # 2. Response (ALL electrodes - Multi regression)

                resp_trials = np.zeros((len(embedding_trial), len(electrode_list_subject), 7))
                for idx_electrode, electrode in enumerate(electrode_list_subject):
                    resp_trial = np.load(os.path.join(load_ND_path,
                                                      "Trial_{}_unattended_response_subject_{}_electrode_{}.npy".format(
                                                          trial_idx + 1, float(subject), electrode)))

                    resp_trials[:, idx_electrode, :] = resp_trial
                resp_trials_list.append(resp_trials)

                # assert(embedding_trials[trial_idx].shape[0] == resp_trials_list[trial_idx].shape[0])
                np.save(
                    "data_formatted_regression/Subject_{}/Layer_{}/Trial_{}_embedding.npy".format(
                        subject, layer_number, trial_idx), embedding_trials[trial_idx])
                np.save(
                    "data_formatted_regression/Subject_{}/Trial_{}_response.npy".format(
                     subject, trial_idx), resp_trials_list[trial_idx])


