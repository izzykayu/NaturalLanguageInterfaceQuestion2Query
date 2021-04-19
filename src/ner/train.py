import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle
import torchwordemb
import torch.backends.cudnn as cudnn
import numpy as np
 
torch.manual_seed(1)

from model import *
from utils import *

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
cudnn.benchmark = True
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, vec).cuda()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-4)
# Check predictions before training
#precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#precheck_tags = torch.cuda.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
#print(model(precheck_sent))
hist_loss = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
origin_total = 0
pred_total = 0
intersection_total = 0
list_output = []
pred_list = []
orig_list = []

for epoch in range(10):  
    epoch_loss = 0
    i = 1
    for sentence, tags in training_data:
        model.zero_grad()
        #sentence_in = torch.cuda.LongTensor([word_to_ix[s] for s in sentence])
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.cuda.LongTensor([tag_to_ix[t] for t in tags])
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.cpu().numpy()
        if i % 1000 ==0:
            print(i)
        i += 1
    for sentence, tags in training_data[6000:]:
        precheck_sent = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(precheck_sent)
        var = autograd.Variable(torch.FloatTensor(tag_scores[1]))
        pred = scores_to_tags(var, tag_to_ix)
        pred_total += len(pred)
        if len(pred) > 0:
            list_output.append('\n'.join(pred))
	tags_var = torch.FloatTensor([tag_to_ix[t] for t in tags])
        tags_var = autograd.Variable(tags_var)
        origin = scores_to_tags(tags_var, tag_to_ix)
        pred_list.append(pred)
        orig_list.append(origin)
   
    avg_loss = epoch_loss / len(training_data)
    print(epoch, avg_loss)
    hist_loss.append(avg_loss)
    acc, prec, recall, fmeasure = get_ner_fmeasure(orig_list,pred_list)
    print("Accuracy:{0}, Precision:{1}, Recall:{2}, f measure:{3}".format(acc,prec,recall,fmeasure))

print(hist_loss)

torch.save(model, './model_10.pth')
