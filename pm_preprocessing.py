# The code produces a sample of 1000 plots. If you would like more, please change the amount_to_save variable in line 165
# 0 - Import Libraries1
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
import os
import csv


# 0.1 - Import Custom Libraries
## No custom library


# 1 - Custom Functions

def fake_paragraph(body):
    paragraphs = body.strip().split('LUcount')
    i = 0
    new_body = ""
    for p in paragraphs:
        new_body = new_body + p
        if i == int(len(paragraphs)*15/100):
            new_body = new_body + " <p> "
        if i == int(len(paragraphs)*45/100):
            new_body = new_body + " <p> "
        if i == int(len(paragraphs)*75/100):
            new_body = new_body + " <p> "
        if i == len(paragraphs):
            new_body = new_body + " <p> "
        i=i+1
    return new_body

def sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2

def trim_body(body):
    paragraphs = body.replace(' <p> ', '\n').split('\n')
    body_new = []
    par_length = 1

    for par in paragraphs:
        _par = par
        if _par.endswith(' <s>.'):
            _par = _par[:-5]
        temp_body = _par.replace(' <s>', ' ').replace('  ', ' ').strip()
        sentences = _par.replace(' <s>', '\n').replace('  ', ' ').strip().split('\n')
        if len(paragraphs) == 1:
            s = 0
            first = True
            while len(sentences[s].split(' ')) < 4 or '::act ' in sentences[s].lower() or ' act:' in sentences[s].lower():
                s+=1
                if s == len(sentences):
                    return None
            body_new.append('<o> ' + sentences[s].replace(' <s> ', ' ').strip())
            s+=1

            while s < len(sentences) and len(body_new)< 5:
                body_new.append('<o>')
                curr_len = 0
                while s < len(sentences) and curr_len + len(sentences[s].split(' ')) < 400:
                    if ':act ' in sentences[s].lower() or 'act: ' in sentences[s].lower() :
                        s+=1
                        break

                    if len(sentences[s]) > 10:
                        curr_len += len(sentences[s].replace(' <s> ', ' ').strip().split(' '))
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s += 1

        else:
            if par_length >5:
                s = 0
                while s < len(sentences) and len(sentences[s]) > 10 and (len(body_new[len(body_new)-1].split(' ')) + len(sentences[s].split(' '))) < 400:
                    if len(sentences[s]) > 10:
                        body_new[len(body_new) - 1] += " " + sentences[s].replace(' <s> ', ' ').strip()
                        body_new[len(body_new) - 1] = body_new[len(body_new) - 1].strip()
                    s+=1
            else:
                if len(temp_body) > 10 and len(temp_body.split(' ')) <= 400:
                    body_new.append(temp_body.replace(' <s>', ' ').replace('  ', ' ').strip())

                elif len(temp_body.split(' ')) >400:
                    curr_len  = 0
                    newstr = ''
                    for sent in sentences:
                        x = (' ' + sent).strip()
                        if len(newstr.split(' ')) + len(sent.split(' ')) <= 400:
                            newstr += (' '+ sent).strip()
                        else:
                            newstr += (' '+ sent).strip()
                            newstr = newstr[0:400]
                    body_new.append(newstr.replace(' <s>', ' ').replace('  ', ' ').strip())

        par_length+=1


    return body_new

def clean_top_features(keywords, top=10):
    keywords = sorting(keywords)
    newkeys = []
    newkeys.append(keywords[len(keywords)-1])
    for i in range(len(keywords)-2,-1,-1):
        if newkeys[len(newkeys)-1].startswith(keywords[i]):
            continue
        newkeys.append(keywords[i])

    if len(newkeys) > top:
        return newkeys[:10]
    return newkeys

def convert_keys_to_str(key_list):
    newstr = key_list[0]
    for k in range(1, len(key_list)):
        if len(key_list[k].split(' ')) > 2 :
            newstr += '[SEP]' + key_list[k]
    return newstr.replace("(M)", "").strip()

# 2 - Run Code

# 2.1 - Inputs & Outputs

# Input : wikiplots file ( plots, titles)
## Sample downloaded from https://github.com/markriedl/WikiPlots (there is a plots.zip folder )
## Sample input:
    ## plots : Old Major, the old boar on the Manor Farm, summons the animals on the farm together for a meeting, during which he refers to humans as "enemies" and teaches the animals a revolutionary song called "Beasts of England".
    ## titles: Animal Farm
# Output: wikiplot.kwRAKE.csv
# Sample output:
## plot-1_0	K	animal farm[SEP]happiest animals live simple lives .'[SEP]several men attack animal farm .'[SEP]napoleon educates young puppies[SEP]boxer continues working harder[SEP]irresponsible farmer mr jones[SEP]set aside special food items[SEP]frequently smears snowball[SEP]anthem glorifying napoleon[SEP]revolutionary song called[SEP]similar animal revolts .'	I	4	Old Major, the old boar on the Manor Farm, summons the animals on the farm together for a meeting, during which he refers to humans as "enemies" and teaches the animals a revolutionary song called "Beasts of England". When Major dies, two young pigs, Snowball and Napoleon, assume command and consider it a duty to prepare for the Rebellion. The animals revolt and drive the drunken and irresponsible farmer mr Jones from the farm, renaming it "Animal Farm". They adopt the Seven Commandments of Animalism, the most important of which is, "All animals are equal". Snowball teaches the animals to read and write, while Napoleon educates young puppies on the principles of Animalism. Food is plentiful, and the farm runs smoothly.	NA


infile = 'data/download/plots'
infile_title = 'data/download/titles'
outfile = 'data/generated/wikiplot.kwRAKE.csv'


# 2.2 - Execute
r = Rake()
vectorizer = TfidfVectorizer(ngram_range=(1,3))
topK = 10

f = open(infile, 'r', encoding='"ISO-8859-1"')
f_title = open(infile_title, 'r', encoding='"ISO-8859-1"')
fout = open(outfile, 'a', encoding='"ISO-8859-1"')

lines = f.readlines()
lines_title = f_title.readlines()

abstract_lens = {}

print("Starting Pre-processing")
sentences_to_write = []
w = 0
total = 0
sentences_to_write.append("[ID]\t[KEY/ABSTRACT]\t[KEYWORDS]\t[DISCOURSE (T/I/B/C)]\t[NUM_PARAGRAPHS]\t[PARAGRAPH]\t[PREVIOUS_PARAGRAPH]\n")

title_id = 0
new_line =0
document = []
body = ""
amount_to_save = 1000 #just to make a sample
saved = 0
for l in range(len(lines)):
    if lines[l].strip().startswith("<EOS>"):
        document.append(str(lines[l].replace('t outline . <s>', '').replace(' <p> ', ' ').replace('  ', ' ').strip().replace(' <s> ', '\n').split('\n')).strip('[]'))
        body = body + " " + str(lines[l].replace('t outline . <s>', '').replace(' <p> ', ' ').replace('  ', ' ').strip().replace(' <s> ', '\n').split('\n')).strip('[]')
        title = lines_title[title_id].strip()
        title_id+=1
        r = Rake()
        r.extract_keywords_from_sentences(document)
        top_features = r.get_ranked_phrases()
        top_features = clean_top_features(top_features, topK)
        keywordsSTR = convert_keys_to_str(top_features)
        if len(title) > 2:
            title = title.lower().replace("paid notice :", "").replace("paid notice:", "").replace("journal;",
                                                                                                   "").strip()
            keywordsSTR = title + '[SEP]' + keywordsSTR
            if len(keywordsSTR.split(' ')) > 100:
                keywordsSTR = ' '.join(keywordsSTR.split(' ')[0:100]).strip()

        fake_body =fake_paragraph(body)
        body_new = trim_body(fake_body)

        if len(body_new)  > 1: # We ignore all with a body less than 1
            id = 'plot-' + str(title_id)
            total += 1
            new_sentence = id + '_0\tK\t' + keywordsSTR + '\tI\t' + str(len(body_new)) + "\t" + body_new[0] + "\tNA"
            sentences_to_write.append(new_sentence + '\n')
            for d in range(1, len(body_new) - 1):
                new_sentence = id + '_' + str(d) + '\tK\t' + keywordsSTR + '\tB\t' + str(len(body_new)) + "\t" + body_new[
                    d] + "\t" + body_new[d - 1]
                sentences_to_write.append(new_sentence + '\n')
            if len(body_new) > 1:
                new_sentence = id + '_' + str(len(body_new) - 1) + '\tK\t' + keywordsSTR + '\tC\t' + str(
                    len(body_new)) + "\t" + body_new[len(body_new) - 1] + "\t" + body_new[len(body_new) - 2]
                sentences_to_write.append(new_sentence + '\n')
        document.clear()
        body = " "
        new_line =0
        saved = saved + 1
        if saved == amount_to_save:
            l = len(lines)+1
            break


    else:
        document.append(str(lines[l].replace('t outline . <s>', '').replace(' <p> ', ' ').replace('  ', ' ').strip().replace(' <s> ', '\n').split('\n')).strip('[]'))
        body = body +" "+ "".join(lines[l].replace('t outline . <s>', '').replace(' <p> ', ' ').replace('  ', ' ').strip().replace(' <s> ', '\n').split('\n'))
        body = body +'LUcount'


fout.writelines(sentences_to_write)
print("Pre-processing has ended")

# 2.3 - Train - Test - Validation Split
# Input:
    ##wikiplots.kwRAKE.csv (created in step 2.2)
    ## authors original train/test/validation split "wikiplots_splits.txt"
    ## Sample of this input: plot-702	picnic at hanging rock (novel)	train

paper_split = "data/download/wikiplots_splits.txt"

inputfile = 'data/generated/wikiplot.kwRAKE.csv'
train_outfile = 'data/generated/train_encoded.csv'
val_outfile = 'data/generated/val_encoded.csv'
test_outfile = 'data/generated/test_encoded.csv'

print("Starting train/test/validation split")
dicts = {}
i= 0

with open(paper_split, newline='') as body:
    titles = csv.reader(body, delimiter='\t')
    for lines in titles:
        if(len(lines) > 1):
            dicts[lines[1]] = lines[2]





f = open(inputfile, 'r', encoding='"ISO-8859-1"')
train_out = open(train_outfile, 'a', encoding='"ISO-8859-1"')
val_out = open(val_outfile, 'a', encoding='"ISO-8859-1"')
test_out = open(test_outfile, 'a', encoding='"ISO-8859-1"')

train_to_write = []
va_to_write=[]
test_to_write=[]

lines = f.readlines()

for l in range(len(lines)):
    title = lines[l].split('\t')[2]
    title = title.split('[SEP]')[0]
    try:
        result = dicts[title]
        if result == 'train':
            train_to_write.append(lines[l])
        elif result == 'test':
            test_to_write.append(lines[l])
        elif result == 'dev':
            va_to_write.append(lines[l])
    except:
        continue


train_out.writelines(train_to_write)
val_out.writelines(va_to_write)
test_out.writelines(test_to_write)

print("Train/test/validation split has ended")
print("Thank you for your patience")
