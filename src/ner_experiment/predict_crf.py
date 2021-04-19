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

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    #print(idx.data.numpy())
    return idx.data.tolist()[0]


def prepare_sequence(seq, to_ix):
    #idxs = [to_ix[w] for w in seq]
    idxs = [to_ix[w] if w in to_ix.keys() else to_ix['UNKNOWN'] for w in seq]
    tensor =  torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
#         score = torch.zeros(1)
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            #print(self.transitions[tags[i + 1], tags[i]])
            #print(feat[tags[i + 1]])
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    
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
        print(util.tags_to_i2b2(j + 1, sentence, label))
        origin = util.tags_to_i2b2(j + 1, sentence, label)
        origin_total += len(origin)
        
        if sentence == []:
            continue
        # lowercase each word in sentence
        tmp = []
        for w in sentence:
            tmp.append(w.lower())
        sentence = tmp
        
        inputs = prepare_sequence(sentence, word_to_ix)
        score, tag_idxs = model(inputs)

        #print(tag)

        #print(tag_scores[1])
#         print(tag_idxs)
        var = autograd.Variable(torch.FloatTensor(tag_idxs))
        tags = [[key for key in tag_to_ix.keys() if tag_to_ix[key] == idx][0] for idx in tag_idxs]
        
        pred = util.tags_to_i2b2(j + 1, test_sentences[j], tags)
        
        
        # only problem, test and treatment
#         temp = []
#         for p in pred:
#             if 't="problem"' in p or 't="test"' in p or 't="treatment"' in p:
#                 temp.append(p)
#         pred = temp
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
    
    print(intersection_total, pred_total, origin_total)
    precision = intersection_total / pred_total
    recall = intersection_total / origin_total
    print("precision:{0:.2f}".format(precision))
    print("recall:{0:.2f}".format(recall))
    print("f measure:{0:.2f}:".format(2 * precision * recall / (precision + recall)))
    
if __name__ == '__main__':
    with open('data/2012_training_data_py3.pickle', 'rb') as f:
        training_data = pickle.load(f)
    f.close()
    temp = []
    for i in range(len(training_data)):
        temp.append(([w.lower() for w in training_data[i][0]], training_data[i][1]))

    training_data = temp    
    word_to_ix = {}
    char_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    word_to_ix['UNKNOWN'] = len(word_to_ix)
    #print(word_to_ix)
    #print(char_to_ix)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = { 'O':0,

           'B-problem':1, 'B-test':2, 'B-treatment':3, 'B-occurrence':4, 'B-clinical_dept':5, 'B-evidential':6,

           'I-problem':7, 'I-test':8, 'I-treatment':9, 'I-occurrence':10, 'I-clinical_dept':11, 'I-evidential':12,
             START_TAG: 13, STOP_TAG: 14

         }
    
    model = torch.load(sys.argv[3])
    txt = glob.glob(sys.argv[1])
    con = glob.glob(sys.argv[2])
    txt = sorted(txt)
    con = sorted(con)
    for i in range(len(txt)):
        predict(model, txt[i], con[i], '.')