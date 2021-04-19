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

print(training_data[:10])
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

#model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, vec).cuda()
model = torch.load("model_10.pth")

origin_total = 0
pred_total = 0
intersection_total = 0
list_output = []
pred_list = []
orig_list = []

for sentence, tags in training_data[:10]:
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
    origin_total += len(origin)
    intersection_total += len(set(origin) & set(pred))
    pred_list.append(pred)
    orig_list.append(origin)
print(pred_list)
print('')
print(orig_list)
    
precision = intersection_total / pred_total
recall = intersection_total / origin_total
if (precision * recall)==0:
    f_score = 0
else:
    f_score = (2 * precision * recall) / (precision + recall)


print("precision:{0:.2f}".format(precision))
print("recall:{0:.2f}".format(recall))
print("f measure:{0:.2f}".format(f_score))
acc, prec, recall, fmeasure = get_ner_fmeasure(orig_list,pred_list)
print("Accuracy:{0}, Precision:{1}, Recall:{2}, f measure:{3}".format(acc,prec,recall,fmeasure))


