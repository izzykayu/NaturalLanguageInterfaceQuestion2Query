import glob
import os
import re
import numpy as np

def file_to_list(txt_file, con_file):
    '''
    Argv:
        txt_filepath:
        con_filepath
    Return:
        sentences:
        labels:
    '''
    #txt_file = glob.glob(txt_filepath)
    #con_file = glob.glob(con_filepath)
    #print(txt_file)
    with open(txt_file) as tf:
        content = tf.read()
        lines = content.split("\n")
        sentences = [line.split() for line in lines]
        labels = [["O" for w in list_word] for list_word in sentences] 
    tf.close()

    list_groups = []
    with open(con_file) as cf:
        for line in cf.readlines():
            if not line.strip():
                continue
            concept_regex = '^c="(.*)" (\d+):(\d+) (\d+):(\d+)\|\|t="(\w*)"'
            match = re.search(concept_regex, line.strip())
            if match:
                groups = match.groups()
                list_groups.append([int(groups[1]) - 1, int(groups[2]), int(groups[4]), groups[5]])
    cf.close()
    
    for entity in list_groups:
        #print(entity)
        idx_line = entity[0]
        start = entity[1]
        end = entity[2]
        label = entity[3]
        #print(labels[idx_line])
        labels[idx_line][start] = "B-" + label
        if start != end:
            for j in range(start+1, end+1):
                labels[idx_line][j] = "I-" + label
    return sentences, labels


def scores_to_tags(scores, tag_to_ix):
    '''
    function to transform scores for one sentence to tags for this sentence
    Args:
        scores: n*d torch variable, n is the number sequence length, d is the tag numbers
        tag_to_ix: dict that converts a tag to a integer distinctly
    Returns:
        tags: tag list of length n
    '''
    scores = scores.data.numpy()
    tags_idxs = [np.argmax(score) for score in scores]
    tags = [[key for key in tag_to_ix.keys() if tag_to_ix[key] == idx][0] for idx in tags_idxs]
    return tags

def tags_to_i2b2(idx, sentence, tags):
    '''
    Args:
        idx: line index
        sentence: a list of words in the sentence
        tags: a list of tags in 'IOB' format of the sentence
    Return:
        i2b2_list: a list of all the entities recognized in i2b2 format
    '''
    i2b2_list = []
    start = -1
    end = start
    label = None
    #tags = scores_to_tags(scores, tag_to_ix)
    for i in range(len(tags)):
        if 'B' in tags[i]:
            start = i
            end = i
            label = tags[i][2:]
            if i == len(tags) - 1:
                i2b2_list.append('c="{0}" {1}:{2} {1}:{3}||t="{4}"'.format(sentence[start], idx, start, end, label))
        elif 'I' in tags[i]:
            end = i
            if i == len(tags) - 1:
                i2b2_list.append('c="{0}" {1}:{2} {1}:{3}||t="{4}"'.format((' ').join(sentence[start:end+1]), idx, start, end, label))
        else:
            if start > -1 and end >= start:
                i2b2_list.append('c="{0}" {1}:{2} {1}:{3}||t="{4}"'.format((' ').join(sentence[start:end+1]), idx, start, end, label))
            start = -1
            end = start
            label = None
    return i2b2_list

    
def evaluate(con_file, pred_file):
    '''
    Args:
        con_file: one .con gold file
        pred_file: one prediction .con file associated with the gold file
    '''
    with open(con_file, 'r') as cf:
        origin = cf.read().split('\n')
        cf.close()
    with open(pred_file, 'r') as pf:
        pred = pf.read().split('\n')
        pf.close()
    
#     for i in range(len(origin)):
#         print(origin[i])
#         print(pred[i])
        
    # only problem, test and treatment
    temp = []
    for p in pred:
        if 't="problem"' in p or 't="test"' in p or 't="treatment"' in p:
            temp.append(p)
    pred = temp
    
        
    origin_total = len(origin)
    pred_total = len(pred)
    intersection_total = len(set(origin) & set(pred))
    precision = intersection_total / pred_total
    recall = intersection_total / origin_total
    print("precision:{0:.2f}".format(precision))
    print("recall:{0:.2f}".format(recall))
    print("f measure:{0:.2f}:".format(2 * precision * recall / (precision + recall)))
    return origin_total, pred_total, intersection_total