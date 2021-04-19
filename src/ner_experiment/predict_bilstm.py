import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util
import pickle
import os
import sys
import glob

class BiLSTM(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_size, tagset_size):
        super(LSTMTagger_Enhanced, self).__init__()
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
    
def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    for w in seq:
        if w not in to_ix.keys() and w.lower() not in to_ix.keys():
            print(w)
    idxs = [to_ix[w] if w in to_ix.keys() else to_ix[w.lower()] if w.lower() in to_ix.keys() else to_ix['UNKNOWN'] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def prepare_sequence_char(seq, to_ix, char_to_ix):
    idxs = [to_ix[w] if w in to_ix.keys() else to_ix[w.lower()] if w.lower() in to_ix.keys() else to_ix['UNKNOWN'] for w in seq]
    word_tensor = torch.LongTensor(idxs)
    return autograd.Variable(word_tensor), [autograd.Variable(torch.LongTensor([char_to_ix[c] if c in char_to_ix.keys() else char_to_ix['UNKNOWN'] for c in w.lower()])) for w in seq]

def predict(model, txt_file, con_file, output_dir):
    '''
    Args:
        model: model saved by torch.save() command
        txt_file: one .txt file path
        con_fil: one .con file path
        output_dir: directory for store the outcome .con file
    
    '''
    test_sentences, test_labels = util.file_to_list(txt_file, con_file)
    origin_total = 0
    pred_total = 0
    intersection_total = 0
    list_output = []
    for j in range(len(test_sentences)):
        print(j)
        sentence = test_sentences[j]
        label = test_labels[j]
        print(sentence)
        #print(label)
        #print(label)
        print(util.tags_to_i2b2(j, sentence, label))
        origin = util.tags_to_i2b2(j + 1, sentence, label)
        origin_total += len(origin)
        
        if sentence == []:
            continue
        inputs, inputs_char = prepare_sequence_char(sentence, word_to_ix, char_to_ix)
        tag_scores = model(inputs, inputs_char)
        #print(util.scores_to_tags(tag_scores, tag_to_ix))
        #print(util.tags_to_i2b2(j + 1, sentence, util.scores_to_tags(tag_scores, tag_to_ix)))
        pred = util.tags_to_i2b2(j + 1, sentence, util.scores_to_tags(tag_scores, tag_to_ix))
        
        # only problem, test and treatment
        temp = []
        for p in pred:
            if 't="problem"' in p or 't="test"' in p or 't="treatment"' in p:
                temp.append(p)
        pred = temp
        print(pred)
        pred_total += len(pred)
        if len(pred) > 0:
            list_output.append('\n'.join(pred))
        
        print(set(origin) & set(pred))
        intersection_total += len(set(origin) & set(pred))
        print("\n")
    
    basename = os.path.basename(txt_file).split('.')[0]
    output_path = os.path.join(output_dir, basename + '.con')
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w') as f:
        f.write('\n'.join(list_output))
    f.close()
        
    precision = intersection_total / pred_total
    recall = intersection_total / origin_total
    print("precision:{0:.2f}".format(precision))
    print("recall:{0:.2f}".format(recall))
    print("f measure:{0:.2f}:".format(2 * precision * recall / (precision + recall)))
    
if __name__ == '__main__':
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
    word_to_ix['UNKNOWN'] = len(word_to_ix)
    char_to_ix['UNKNOWN'] = len(char_to_ix)
    #print(word_to_ix)
    #print(char_to_ix)
    tag_to_ix = { 'O':0,

               'B-problem':1, 'B-test':2, 'B-treatment':3, 'B-occurrence':4, 'B-clinical_dept':5, 'B-evidential':6,

               'I-problem':7, 'I-test':8, 'I-treatment':9, 'I-occurrence':10, 'I-clinical_dept':11, 'I-evidential':12,

             }
    
    model = torch.load(sys.argv[3])
    txt = glob.glob(sys.argv[1])
    con = glob.glob(sys.argv[2])
    txt = sorted(txt)
    con = sorted(con)
    for i in range(len(txt)):
        predict(model, txt[i], con[i], '.')