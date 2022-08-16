
#%% Load packages

import matplotlib.pyplot as plt
# import os
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

# Import training xml file
data_path = "data/train_texts.xml"
root = parse_xml_file(data_path)

#%% Convert xml to list of documents (list of list of sentences)
docs = parse_xml_file_to_text(root)

### Get a snippet of the data for building proof of concept
docs_demo = docs[0: 9]

#%%
cleaned_docs = []

for doc in docs_demo:

    cleaned_doc = []
    for sentence in doc:
        cleaned_doc.append(clean_text(sentence))

    cleaned_docs.append(cleaned_doc)

#%%
with open("data/docs_demo_processed.pickle", 'wb') as f:
    pickle.dump(cleaned_docs, f)



#####
#%% Join doc in one paragraph + replace newlines
docs_paragraph = [" ".join(doc) for doc in docs]
docs_paragraph = [re.sub(r'\s+', ' ', d.replace('\n', ' ')).strip() for d in docs_paragraph]

# Clean data
docs_cleaned = [clean_text(d) for d in docs_paragraph]

#%% Write data
with open("data/train_text_preprocessed.txt", 'w', encoding='utf-8') as f:
    for d in docs_cleaned:
        f.writelines(d)

#%%
def parse_xml_file(file_path):
    import xml.etree.ElementTree as ET

    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(file_path, parser=xmlp)
    root = tree.getroot()
    return(root)

def parse_xml_file_to_text(root):
    docs = []

    for doc_id in range(len(root)):
        doc_segs = []
        doc = root[doc_id]
        for seg in doc.iter('seg'):
            doc_segs.append(seg.text)
        docs.append(doc_segs)

    return(docs)

def clean_text(text):
    text = text.replace('!', '.')
    text = text.replace(':', ',')
    text = text.replace('--', ',')
    
    reg = "(?<=[a-zA-Z])-(?=[a-zA-Z]{2,})"
    r = re.compile(reg, re.DOTALL)
    text = r.sub(' ', text)
    
    text = re.sub(r'\s-\s', ' , ', text)
    
#     text = text.replace('-', ',')
    text = text.replace(';', '.')
    text = text.replace(' ,', ',')
    text = text.replace('♫', '')
    text = text.replace('...', '')
    text = text.replace('.\"', ',')
    text = text.replace('"', ',')

    text = re.sub(r'--\s?--', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r',\s?,', ',', text)
    text = re.sub(r',\s?\.', '.', text)
    text = re.sub(r'\?\s?\.', '?', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\s+\?', '?', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\.[\s+\.]+', '. ', text)
    text = re.sub(r'\s+\.', '.', text)
    
    return text.strip().lower()
# %%
