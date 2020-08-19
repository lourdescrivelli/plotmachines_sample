# Code to generate the pkl preprocessing
# Note: This process risks of generating CUDA out of memory.
## Need to be careful about:
    # 1) Line  86 - Generation Length : The bigger, the more memory is used
    # 2) Line 114 - How much of the previous paragraph will be encoded
## Since this is a sample code, we have fixed both to 10, but this size is too small for PlotMachines.


# 0 - Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import argparse
import os
import random
import numpy as np
#import rouge
import torch
from torch import nn
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import *
import csv
import pickle




# 0.1 - Import Custom Libraries
## We need the original model definition , look for model.py in the model folder of the original plotmachines code

from model import GPT2BaseModel

## 1 - Custom functions

def tfmclassifier(textlines, model, tokenizer, gen_len):  #From the original code
    '''Create encoding of the previous paragraph (textlines) using the model and tokenizer'''
    clf = []
    nb = len(textlines)
    # if nb < 8:
    wds = torch.zeros(nb, gen_len, dtype=torch.long).cuda()
    mask = torch.zeros(nb, gen_len, dtype=torch.long).cuda()
    for j in range(nb):
        #print("j",j, textlines[j])
        temp = torch.tensor(tokenizer.encode(textlines[j], add_special_tokens=False)[:gen_len])
        wds[j, :len(temp)] = temp.cuda()
        mask[j, :len(temp)] = torch.ones(len(temp), dtype=torch.long).cuda()
    model.eval()
    outputs = model(wds)
    total = (mask.unsqueeze(2).type_as(outputs[0]) * outputs[0]).sum(dim=1) / mask.type_as(outputs[0]).sum(
        dim=1).unsqueeze(1)
    #wds = None
    #mask = None
    #temp = None
    #outputs = None
    return total

def debug_memory(): #Stackoverflow
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

# 2 - Execute

#2.1 - Input & Output
## Input    Our train/test/validation encoded generated in pm_preprocessing
## Output   A pkl file.
        ## Important :  First row (header) is just a 0 (as described in the paper=
        ## The other outputs follow this structure
        ### Tuple [ ‘plot_01’, ‘last paragraph text’ , [number_of_words,768])

inputfile = ["data/generated/train_encoded.csv","data/generated/test_encoded.csv","data/generated/val_encoded.csv"]
filename = ["data/generated/train_encoded_gpt2.pkl","data/generated/test_encoded_gpt2.pkl","data/generated/val_encoded_gpt2.pkl"]

#2.2 - Run
print("Starting the generation")
gptclf = GPT2Model.from_pretrained('gpt2')
gptclf.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gptclf.to(device)
gpttok = GPT2Tokenizer.from_pretrained('gpt2')
gen_len = 10  # Hyperparameter: how long we want our generation to be.
                # CUDA eating little monster. Set to 150 as a sample

for split in range(len(inputfile)):
    f = open(inputfile[split], 'r', encoding='"ISO-8859-1"')
    lines = f.readlines()

#index = "PLOT_NUMBER"
#string = "PREVIOUS_PARAGRAPH_STRING"
#vector = [0,0,0,0,0,0]
#x  = tuple([index]+[string]+vector)
#print(x)


# Create the header = 0
    x=0
    outfile = open(filename[split],'wb')
    pickle.dump(x,outfile)
    outfile.close()

    i=0
    for l in range(len(lines)):
        print("Currently encoding line ",[l],"from ", len(lines))
        plot_id = lines[l].split('\t')[0]
        last_paragraph = lines[l].split('\t')[6]
        prevprc = tfmclassifier(last_paragraph[0:10].split(" "), gptclf, gpttok, gen_len)
        new_tuple = (plot_id,last_paragraph.strip(),prevprc)
        outfile=open(filename[split],'ab+')
        pickle.dump(new_tuple,outfile)
        outfile.close()
        new_tuple = None
        torch.cuda.empty_cache()
        i=i+1
    print("Finished processing file ", filename[split])
print("Generation of pickle files has ended")