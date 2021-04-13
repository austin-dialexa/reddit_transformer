#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np
import tqdm

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


# In[2]:


df = pd.read_csv('final_joke.csv')


# In[3]:


df['selftext'] = df['selftext'].str.strip()
df['title'] = df['title'].str.strip()
df = df.replace("[^A-Za-z0-9' ,.:;!?\\-]+" ,'', regex=True)
df = df.replace(r"\t|\n|\r", "", regex=True)
df = df.replace('"', '', regex=True)
df = df.replace('',float("NaN"),)
df = df.dropna(subset = ["selftext"])


# In[4]:


class JokesDataset(Dataset):
    def __init__(self, jokes_path = 'final_joke.csv'):
        super().__init__()

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"
        self.data = pd.read_csv(jokes_path)

        self.data['string'] = self.data['title'] + self.data['selftext']
        for row in self.data['string']:
            joke_str = f"{row}{self.end_of_text_token}"
            self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


# In[5]:


dataset = JokesDataset()
joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# In[6]:


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelWithLMHead.from_pretrained("distilgpt2")
model = model.to(device)


# In[7]:


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


# In[8]:


BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
from transformers import AdamW, get_linear_schedule_with_warmup

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


# In[9]:


model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,joke in enumerate(tqdm(joke_loader)):
        
        #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
        #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if joke_tens.size()[1] > MAX_SEQ_LEN:
            continue
        
        #The first joke sequence in the sequence
        if not torch.is_tensor(tmp_jokes_tens):
            tmp_jokes_tens = joke_tens
            continue
        else:
            #The next joke does not fit in so we process the sequence and leave the last joke 
            #as the start for next sequence 
            if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                work_jokes_tens = tmp_jokes_tens
                tmp_jokes_tens = joke_tens
            else:
                #Add the joke to sequence, continue and try to add more
                tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:,1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################
            
        outputs = model(work_jokes_tens, labels=work_jokes_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_{epoch}.pt"))
            


# In[ ]:




