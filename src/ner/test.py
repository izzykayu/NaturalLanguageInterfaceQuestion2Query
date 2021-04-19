

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim 
from model import *
from utils import *
import pickle
import torchwordemb

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
HIDDEN_DIM = 100

# Load training data
with open("2012_training_data_py2.pickle","rb") as f:
    training_data = pickle.load(f)  

index = []                                   
for i,sent in enumerate(training_data):
    if sent[0]!=[]:
        index.append(i)

training_data = [training_data[i] for i in index]

#print(training_data[:10])
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = { 'O':0,

          'B-problem':1, 'B-test':2, 'B-treatment':3, 'B-occurrence':4, 'B-clinical_dept':5, 'B-evidential':6,

          'I-problem':7, 'I-test':8, 'I-treatment':9, 'I-occurrence':10, 'I-clinical_dept':11, 'I-evidential':12,

          START_TAG:13, STOP_TAG:14

        }

embeddings,vec = torchwordemb.load_word2vec_bin("wikipedia-pubmed-and-PMC-w2v.bin")
EMBEDDING_DIM = vec.size(1)

with open("test_questions.txt") as tf:
    content = tf.read()
    lines = content.split("\n")
    sentences = [line.split() for line in lines]
tf.close()
print(sentences[:10])

#model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, vec).cuda()
#model.load_state_dict(torch.load("model.pth"))
model = torch.load("model_10.pth")
for sentence in sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

pred_total = 0
list_output = []    
for j in range(len(sentences)):
    sentence = sentences[j]
    if sentence == []:
        continue
    sentence = prepare_sequence(sentence, word_to_ix)
    # here the tag_scores are the tags
    score, tag_idxs = model(sentence)
    #print(tag)
    #print(tag_scores[1])
    var = autograd.Variable(torch.FloatTensor(tag_idxs))
    pred = scores_to_tags(var, tag_to_ix)
    print(sentences[j])
    print(tags_to_i2b2(j, sentences[j], pred))
    #pred = tags_to_i2b2(j, sentence, scores_to_tags(var, tag_to_ix))
    pred_total += len(pred)
    if len(pred) > 0:
        list_output.append('\n'.join(pred))
    #if j==(len(sentences)-2):
    #    print(list_output)
'''
output_path = os.path.join("/scratch/as11566/NLU/", 'Testoutput.con')
with open(output_path, 'w') as f:
    f.write('\n'.join(list_output))    
'''
