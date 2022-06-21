#%% Load packages

import matplotlib.pyplot as plt
import os
import json
import pickle
import torch
import numpy as np
import re
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

#%%
model_type = 'bert-base-uncased' #albert-base-v1, bert-base-cased, bert-base-uncased

#%% SETUP TOKENIZER

# Initiate
tokenizer = AutoTokenizer.from_pretrained(model_type)

# Get encode of 3 special punctuation
target_ids = tokenizer.encode(".?,")[1:-1]

# Build dictionary of target tokens
id2target = {
    0: 0,
    -1: -1,
}
for i, ti in enumerate(target_ids):
    id2target[ti] = i+1
target2id = {value: key for key, value in id2target.items()}

'''
Vocab of tokenizer: PERIOD = 1012, QUESTION = 1029, COMMA = 1010
Target code: 0 = followed by SPACE, -1 = subtokens in the middle, 1 = PERIOD, 2 = QUESTION, 3 = COMMA

'''

def create_target(text): # Input paragraph

    target_token2id = {t: tokenizer.encode(t)[-2] for t in ".?,"}
    encoded_words, targets = [], []
    
    words = text.split(' ')

    for word in words:
        target = 0
        for target_token, target_id in target_token2id.items(): 
        # Loop through all special tokens PERIOD COMMA QUESTION
            if word.endswith(target_token):
                word = word.rstrip(target_token)
                target = id2target[target_id] # Change target token id

        encoded_word = tokenizer.encode(word, add_special_tokens=False)
        
        for w in encoded_word:
            encoded_words.append(w)
        for _ in range(len(encoded_word)-1): # If multiple subwords are tokenized than add -1
            targets.append(-1)
        targets.append(target)
        
        assert(len(encoded_word)>0)

    encoded_words = [tokenizer.cls_token_id or tokenizer.bos_token_id] +\
                    encoded_words +\
                    [tokenizer.sep_token_id or tokenizer.eos_token_id]
    targets = [-1] + targets + [-1] # First + last token = -1
    
    return encoded_words, targets