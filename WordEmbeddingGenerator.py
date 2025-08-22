"""
Embedding Generator with Mistral-7B
-----------------------------------

This script generates word-aligned embeddings from text transcripts using
Mistral-7B. Includes options for context length control and removal of 
consecutive word repetitions.

Author: Corentin Puffay
"""

# ===============================
# Imports
# ===============================

# Standard library
import os
import re
from collections import defaultdict

# Third-party
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ===============================
# Helper Functions
# ===============================

def get_indices(words):
    """
    Groups token indices by word.

    Parameters
    ----------
    words : list of tuples
        Each tuple is (word, position, index).

    Returns
    -------
    list of tuples
        Each tuple is (word, [indices]) with 1-based token indices.
    """
    final_result = []
    current_word, current_pos, current_indices = None, None, []

    for word, pos, index in words:
        if not word:
            continue
        if word == current_word and pos == current_pos:
            current_indices.append(index)
        else:
            if current_word is not None:
                final_result.append((current_word, [i + 1 for i in current_indices]))
            current_word, current_pos, current_indices = word, pos, [index]

    if current_word is not None:
        final_result.append((current_word, [i + 1 for i in current_indices]))

    return final_result


def match_pos(full_transcript, text_only):
    """
    Matches positions of words from one transcript against another.

    Parameters
    ----------
    full_transcript : str
        Full text transcript.
    text_only : str
        Cleaned text (subset).

    Returns
    -------
    list of tuples
        Each tuple is (word_upper, start_position).
    """
    cleaned_original = full_transcript.lower()
    cleaned_capitalized = text_only.lower()

    results, start_pos = [], -1
    for word in cleaned_capitalized.split():
        word_len = len(results[-1][0]) if results else 2
        start_pos_new = cleaned_original.find(word, start_pos + word_len - 1)
        if start_pos_new == -1:
            raise ValueError(f"Word '{word}' not found in transcript.")
        start_pos = start_pos_new
        results.append((word.upper(), start_pos))
    return results


def get_words(tks, pos, full_transcript, text_only, tokenizer):
    """
    Aligns tokens to words.

    Returns
    -------
    words : list of tuples
        (capitalized_word, position, token_index)
    """
    matched_pos = match_pos(full_transcript, text_only)
    words, j = [], 0

    for i in range(len(pos)):
        if i == len(pos) - 1:
            words.append(('', matched_pos[-1][1], i))
            continue

        matched = matched_pos[j]
        cap_word, cap_pos = matched
        token = tokenizer.decode([tks[i]]).lower()

        if token in cap_word.lower():
            words.append((cap_word, cap_pos, i))
        else:
            words.append(('', cap_pos, i))

        if j < len(matched_pos) - 1:
            next_cap_word, next_cap_pos = matched_pos[j + 1]
            if pos[i + 1][0] + 1 >= next_cap_pos:
                j += 1

    assert j + 1 == len(matched_pos)
    return words


def remove_consecutive_repetitions_with_indices(text):
    """
    Removes consecutive repeated words and returns indices of removed words.
    """
    pattern = r'\b(\w+)\s+\1\b'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    indices = [
        len(re.findall(r'\b\w+\b', text[:match.start()])) + 1
        for match in matches
    ]
    cleaned_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    return cleaned_text, indices


# ===============================
# Main Embedding Function
# ===============================

def generate_embedding_mistral(transcript, list_idx_duplicated, context_len, full, idx_trial):
    """
    Generates embeddings from a transcript using Mistral-7B.

    Parameters
    ----------
    transcript : str
        Input transcript text.
    list_idx_duplicated : list
        List of indices of duplicated tokens.
    context_len : int
        Context length in tokens.
    full : {"full", "reduced"}
        Whether to use full context or reduced context.
    idx_trial : int
        Trial index (for saving results).

    Returns
    -------
    dict
        Dictionary with hidden states, logits, and alignments.
    """
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", device_map="auto", offload_folder="save_folder", force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", unk_token="<unk>", bos_token="<s>", eos_token="</s>", use_fast=True, force_download=True
    )


    inputs = tokenizer(transcript, return_offsets_mapping=True)
    tks, pos = inputs['input_ids'][1:], inputs['offset_mapping']
    words = get_words(tks, pos, transcript, transcript, tokenizer)
    words_tokens = get_indices(words)

    truncated_inputs = {
        'input_ids': torch.LongTensor(inputs['input_ids']).unsqueeze(0),
        'attention_mask': torch.LongTensor(inputs['attention_mask']).unsqueeze(0),
    }

    if full == "reduced":
        hidden_states_alltoks, logits_alltoks = [], []
        for tk_idx in tqdm(range(1, truncated_inputs['input_ids'].shape[-1] + 1)):
            first_tk_idx = max([0, tk_idx - context_len])
            truncated_inputs_contextlen = {
                'input_ids': truncated_inputs['input_ids'][:, :tk_idx],
                'attention_mask': torch.ones_like(truncated_inputs['attention_mask'][:, :tk_idx]),
            }
            truncated_inputs_contextlen['attention_mask'][:, :first_tk_idx] = 0
            with torch.no_grad():
                outputs = model(**truncated_inputs_contextlen, output_hidden_states=True)
            if not hidden_states_alltoks:
                hidden_states_alltoks = [[] for _ in outputs['hidden_states']]
            for layer, hs in enumerate(outputs['hidden_states']):
                hidden_states_alltoks[layer].append(hs[:, -1, :].cpu().numpy())
            logits_alltoks.append(outputs['logits'][:, -1, :].cpu().numpy())

        hidden_states_alltoks = [np.concatenate(hh, axis=0) for hh in hidden_states_alltoks]
        logits_alltoks = np.concatenate(logits_alltoks, axis=0)

        tokens_kept_idx = [tokens_per_word[-1] for _, tokens_per_word in words_tokens]
        ds_embedding = tuple(np.array([hidden_states_alltoks[k][tokens_kept_idx, :] for k in range(len(hidden_states_alltoks))]))
        ds_embedding_corrected = tuple(np.array([np.delete(ds_embedding[k], list_idx_duplicated, axis=0) for k in range(len(ds_embedding))]))

        return {
            'hidden_states': ds_embedding_corrected,
            'logits': logits_alltoks,
            'alignments': words_tokens,
        }

    else:  # full context
        with torch.no_grad():
            outputs = model(**truncated_inputs, output_hidden_states=True)

        hidden_states = tuple(hs.numpy().squeeze() for hs in outputs['hidden_states'])
        logits = outputs['logits'].numpy().squeeze()

        tokens_kept_idx = [tokens_per_word[-1] for _, tokens_per_word in words_tokens]
        ds_embedding = tuple(np.array([hidden_states[k][tokens_kept_idx, :] for k in range(len(hidden_states))]))
        ds_embedding_corrected = tuple(np.array([np.delete(ds_embedding[k], list_idx_duplicated, axis=0) for k in range(len(ds_embedding))]))

        return {
            'hidden_states': ds_embedding_corrected,
            'logits': logits,
            'alignments': words_tokens,
        }


# ===============================
# Script Entry Point
# ===============================

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    # Example usage:
    # result = generate_embedding_mistral("your transcript here", [], 128, "full", 0)
