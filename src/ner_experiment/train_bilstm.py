import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#for own training data
import pickle
import numpy as np
from util import *

torch.manual_seed(1)

class BiLSTM(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.char_hidden = self.init_hidden_char(char_hidden_dim)
        self.lstm = nn.LSTM(char_hidden_dim+word_embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def init_hidden_char(self, hidden_dim):
        return (autograd.Variable(torch.zeros(1, 1, hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, hidden_dim)))
    
    def init_hidden(self, hidden_dim):
        return (autograd.Variable(torch.zeros(2, 1, hidden_dim // 2)),
                autograd.Variable(torch.zeros(2, 1, hidden_dim // 2)))
    
    def forward(self, sentence, charsets):
        charset_lstm_out = []
        for charset in charsets:
            char_embeds = self.char_embedding(charset)
            char_lstm_out, char_hidden = self.char_lstm(
                char_embeds.view(len(charset), 1, -1), self.char_hidden)
            #take the last hidden 
            charset_lstm_out.append(char_lstm_out[-1].view(1, -1))
        charset_lstm_out = torch.cat(charset_lstm_out)
        word_embeds = self.word_embedding(sentence)
        embeds = torch.cat([charset_lstm_out, word_embeds], dim=1)
        lstm_out, hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def prepare_sequence_char(seq, to_ix, char_to_ix):
    idxs = [to_ix[w.lower()] for w in seq]
    word_tensor = torch.LongTensor(idxs)
    return autograd.Variable(word_tensor), [autograd.Variable(torch.LongTensor([char_to_ix[c] for c in w.lower()])) for w in seq]

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

with open('data/2012_training_data_py3.pickle', 'rb') as f:
    training_data = pickle.load(f)
word_to_ix = {}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word.lower()] = len(word_to_ix)
            for c in word:
                if c not in char_to_ix:
                    char_to_ix[c.lower()] = len(char_to_ix)
word_to_ix['UNKONWN'] = len(word_to_ix)
char_to_ix['UNKONWN'] = len(char_to_ix)
print(word_to_ix)
print(char_to_ix)
tag_to_ix = { 'O':0,

           'B-problem':1, 'B-test':2, 'B-treatment':3, 'B-occurrence':4, 'B-clinical_dept':5, 'B-evidential':6,

           'I-problem':7, 'I-test':8, 'I-treatment':9, 'I-occurrence':10, 'I-clinical_dept':11, 'I-evidential':12,

         }

WORD_EMBEDDING_DIM = 200
CHAR_EMBEDDING_DIM = 10
HIDDEN_DIM = 6
CHAR_HIDDEN_DIM = 6

model = BiLSTM(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM, len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# inputs, inputs_char = prepare_sequence_char(training_data[10][0], word_to_ix, char_to_ix)
# tag_scores = model(inputs, inputs_char)
# print(training_data[10][0])
# print(scores_to_tags(tag_scores, tag_to_ix))

print("training started")
hist_loss = []
for epoch in range(5):
    epoch_loss = 0
    i = 1
    for sentence, tags in training_data:
        model.zero_grad()
        model.char_hidden = model.init_hidden_char(CHAR_HIDDEN_DIM)
        model.hidden = model.init_hidden(HIDDEN_DIM)
        if sentence == []:
            continue
        word_t, char_t = prepare_sequence_char(sentence, word_to_ix, char_to_ix)
        
        tag_scores = model.forward(word_t, char_t)
        targets = prepare_sequence(tags, tag_to_ix)
        
        loss = loss_function(tag_scores, targets)
        epoch_loss += loss.data.numpy()
        loss.backward()
        optimizer.step()
        #print(i, loss)
        if i % 1000 == 0:
            print(i, epoch_loss / i)
        i += 1
    avg_loss = epoch / len(training_data)
    print(epoch, avg_loss)
    hist_loss.append(str(avg_loss))
    if epoch % 50 == 0:
        torch.save(model, 'epoch{0}.model'.format(epoch))
        with open('log_epoch{0}.txt'.format(epoch), 'w') as f:
            f.write('\n'.join(hist_loss))
        f.close()