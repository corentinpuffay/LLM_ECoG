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
#from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, LeaveOneOut
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
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
from sklearn.metrics import roc_curve,auc
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import h5py
import hdf5storage
import json
import torch
import torch.nn.functional as F
import tqdm
from joblib import Parallel, delayed


###### END IMPORTS #####


######################################### FUNCTION DEFINITION #########################################

def get_indicies(words):
    from collections import defaultdict

    # Initialize an empty list to store the final result
    final_result = []

    # Initialize a word to None and an empty list to store indices
    current_word = None
    current_pos = None
    current_indices = []

    # Iterate over the word_tuples list
    for word, pos, index in words:
        if word:  # Skip empty words
            if word == current_word and pos == current_pos:
                # If the word is the same as the current word, append the index
                current_indices.append(index)
            else:
                # If the word is different and the current word is not None, append the result
                if current_word is not None:
                    final_result.append((current_word, [i + 1 for i in current_indices]))
                # Update the current word and indices
                current_word = word
                current_pos = pos
                current_indices = [index]

    # Don't forget to add the last group of words
    if current_word is not None:
        final_result.append((current_word, [i + 1 for i in current_indices]))
        
    return final_result


# Text inputs
def match_pos(full_transcript, text_only):

    original_text = full_transcript
    capitalized_text = text_only

    # Clean texts
    cleaned_original = original_text.lower()
    cleaned_capitalized = capitalized_text.lower()

    # Split cleaned capitalized text into words
    cleaned_capitalized_words = cleaned_capitalized.split()

    # Initialize results list and starting position
    results = []
    start_pos = -1
        
    # For each word in the cleaned capitalized text
    for word in cleaned_capitalized_words:
        # Find the first occurrence of the word in the cleaned original text after the last found word
        word_len = 0
        if len(results) != 0:
            word_len = len(results[-1][0])
        else:
            word_len = 2
        start_pos_new = cleaned_original.find(word, start_pos + word_len - 1)
        assert start_pos_new > -1
        start_pos = start_pos_new
        results.append((word.upper(), start_pos))
    
    return results


def get_words(tks, pos, full_transcript, text_only, tokenizer):


    
    matched_pos = match_pos(full_transcript, text_only)
    pdb.set_trace()
    
    curr_pos = 0
    words = []
    j = 0
    total_matched = 0

    for i in range(len(pos)):
        if i == len(pos) - 1:
            words.append(('', cap_pos, i))
            continue

        curr_pos = pos[i+1][0]

        matched = matched_pos[j]
        cap_word = matched[0]
        cap_pos = matched[1]


        if tokenizer.decode([tks[i]]).lower() in cap_word.lower():
            words.append((cap_word, cap_pos, i))
        else:
            words.append(('', cap_pos, i))
                
        if j != len(matched_pos) - 1:
            next_matched = matched_pos[j + 1]
            if i != len(pos) - 2:
                curr_pos = pos[i+2][0]
                next_token = tokenizer.decode([tks[i+1]])
                
                if curr_pos + 1 >= next_matched[-1]:
                    j += 1
                
    assert j + 1 == len(matched_pos)
    
    return words

##################################################################################


######################################### MAIN #########################################

#### 1. Load the model

torch.set_num_threads(os.cpu_count())


def generateEmbeddingMistralDifferentialContext(transcript, list_idx_duplicated, context_len, full, idx_trial):

    '''
    Definition: Generates embedding of any transcripts without punctuation, includes a parameter to vary the context provided to Mistral-7B

    Arguments: 

    - transcript: string containing the words of the transcript
    - list_idx_duplicated: duplicated tokens
    - context_len: how many tokens to give to Mistral to generate the embeddings
    '''

    #print('*'*80)
    #print(f'Context length: {context_len} tokens')
    
    #print("Loading Mistral-7B")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", offload_folder="save_folder")
    #print("Finished")
    
    
    #print("Number of parameters: {}".format(model.num_parameters()))
    
    
    ### Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    
    
    
        
    ##### HERE DELETE REPEATED WORDS (before tokenization)
    import re

    def remove_consecutive_repetitions_with_indices(text):
        # Regular expression to match consecutive repeated words, case insensitive
        pattern = r'\b(\w+)\s+\1\b'

        # Find all matches for consecutive repeated words (case insensitive)
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        # List to store the index (1-based) of the second repeated word in each repetition
        indices = []

        # Loop through each match to find the second word index
        for match in matches:
            # The index of the second word will be the end of the match minus one (for 1-based indexing)
            second_word_index = len(re.findall(r'\b\w+\b', text[:match.start()])) + 1
            indices.append(second_word_index)
            
        #pdb.set_trace()

        # Remove the consecutive repetitions from the text (case insensitive)
        cleaned_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

        return cleaned_text, indices


    # Get cleaned text and indices of the second repeated words
    
    transcript_new, indices = remove_consecutive_repetitions_with_indices(transcript)
    
    pdb.set_trace()

    # Output results
    #print("Cleaned text:", transcript_new)
    #print("Indices of second repeated words:", indices)
    
    
    np.save(f'repeated_words_unattended_last/repeated_word_indices_trial_{idx_trial+1}.npy', np.array(indices))

    

    # Tokenize the full transcript
    inputs = tokenizer(transcript_new, return_offsets_mapping=True)
    tks = inputs['input_ids'][1:]
    pos = inputs['offset_mapping']
    

    words = get_words(tks, pos, transcript_new, transcript_new, tokenizer)
    words_tokens = get_indicies(words)
    
    
    pdb.set_trace()
    
    # Size input: 
    truncated_inputs = dict()
    truncated_inputs['input_ids'] = torch.LongTensor(inputs['input_ids']).unsqueeze(0)
    truncated_inputs['attention_mask'] = torch.LongTensor(inputs['attention_mask']).unsqueeze(0)


    ####################################################################################
    ##################################################################
    ######################################################

    # Context not full, but reduced by value context_len
    if full == "reduced":
        
      #  print('context reduced')

        # Loop through all tokens and truncate the input to context_len
        hidden_states_alltoks = []
        logits_alltoks = []
    
        # Iterate over tokens
        for tk_idx in tqdm.tqdm(range(1, truncated_inputs['input_ids'].shape[-1]+1)):
            
            pdb.set_trace()
    

            # Create a dict 
            truncated_inputs_contextlen = dict()
            # First token idx is 0 if the context_length is longer than current_idx - first_idx
            first_tk_idx = max([0, tk_idx-context_len])

            # Dig there!
            truncated_inputs_contextlen['input_ids'] = truncated_inputs['input_ids'][:,:tk_idx]
            truncated_inputs_contextlen['attention_mask'] = torch.ones_like(truncated_inputs['attention_mask'][:,:tk_idx])
    
            # Set the attention mask to 0 for tokens not contained in the context length.
            truncated_inputs_contextlen['attention_mask'][:,:first_tk_idx] = 0
            
            
            with torch.no_grad():
                outputs = model(**truncated_inputs_contextlen, output_hidden_states=True)
            if len(hidden_states_alltoks) == 0:
                hidden_states_alltoks = [[] for _ in range(len(outputs['hidden_states']))] # empty list for each layer
            for layer in range(len(outputs['hidden_states'])):
                hidden_states_alltoks[layer].append(outputs['hidden_states'][layer][:,-1,:].cpu().numpy().copy()) # shape 1, hidden_dim
            logits_alltoks.append(outputs['logits'][:,-1,:].cpu().numpy().copy()) # shape 1, hidden_dim
            del outputs

        # All layers
        hidden_states_alltoks = [np.concatenate(hh, axis=0) for hh in hidden_states_alltoks]
        logits_alltoks = np.concatenate(logits_alltoks, axis=0)

        output = dict()
        output['hidden_states'] = tuple(hidden_states_alltoks)
        output['logits'] = logits_alltoks
        output['alignments'] = words_tokens

        # Word-align the hidden states
        tokens_kept_idx = []
        for tokens_per_word in output['alignments']:
            tokens_kept_idx.append(tokens_per_word[-1][-1])
        
        # Extracts the activation matrix for a given layer / a given trial
        actv =  output['hidden_states'] # (33, # words, 4096)
        
        # Downsample to last token of each word (per layer)
        ds_embedding = tuple(np.array([actv[k][tokens_kept_idx,:] for k, ele in enumerate(actv)]))
    
        
        # Remove extra embeddings not represented in wrd_label + apply PCA
        ds_embedding_corrected =  tuple(np.array([np.delete(ds_embedding[k], list_idx_duplicated, axis=0) 
                                                  for k, ele in enumerate(ds_embedding)]))
    
            
        # Replace the embedding with the word-aligned one
        output['hidden_states'] = ds_embedding_corrected
    
    
        return output
        


    # Full context, like previously
    else:
        

        with torch.no_grad():
            outputs = model(**truncated_inputs, output_hidden_states=True)
            
            


        out = dict()
        out['hidden_states'] = tuple([t.numpy().squeeze() for t in outputs['hidden_states']])
        out['logits'] = outputs['logits'].numpy().squeeze()
        out['alignments'] = words_tokens
     
    
        # Word-align the hidden states
        tokens_kept_idx = []
        for tokens_per_word in out['alignments']:
            tokens_kept_idx.append(tokens_per_word[-1][-1])
        
        # Extracts the activation matrix for a given layer / a given trial
        actv =  out['hidden_states'] # (33, # words, 4096)
        
        # Downsample to last token of each word (per layer)
        ds_embedding = tuple(np.array([actv[k][tokens_kept_idx,:] for k, ele in enumerate(actv)]))

        # Remove extra embeddings not represented in wrd_label + apply PCA
        ds_embedding_corrected =  tuple(np.array([np.delete(ds_embedding[k], list_idx_duplicated, axis=0) 
                                                  for k, ele in enumerate(ds_embedding)]))
    
            
            
        # Replace the embedding with the word-aligned one
        out['hidden_states'] = ds_embedding_corrected

        try:
            assert(len(transcript.split(" "))-len(indices) == len(transcript_new.split(" ")))
            assert(len(transcript.split(" "))-len(indices) == len(ds_embedding_corrected[1]))
        except:
            pdb.set_trace()
            
            
        return out, len(indices)
        

##################################################################################







#### END MAIN SCRIPT ####