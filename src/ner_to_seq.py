import util
import sys
if __name__ == '__main__':
    sentences, labels = util.file_to_list(sys.argv[1], sys.argv[2])
    #print(sentences[0], labels[0])
    #only exact match for these three entities
    entity_dict = {
        'religion': [
            'CATHOLIC',
            'CHRISTIAN',
            'PROTESTANT QUAKER',
            'METHODIST',
            '7TH DAY ADVENTIST',
            'GREEK ORTHODOX',
            'HINDU',
            'MUSLIM',
            'ROMANIAN', 
            'BUDDHIST',
            'HEBREW',
            'LUTHERAN',
            'BAPTIST',
            "JEHOVAH'S WITNESS",
            'JEWISH'
        ],
        'insurance':[
            'Self Pay',
            'Medicare',
            'Medicaid',
            'Private',
            'Government'
        ],
        'ethinicity':[
            'HISPANIC',
            'BLACK',
            'MIDDLE EASTERN',
            'ASIAN',
            'WHITE'
        ]
    }
    list_out = [[] for s in sentences]
    list_dict = [{} for s in sentences]

    #print(labels)
    for i in range(len(sentences)):
        flag_inlabel = 0
        idx_start = 0
        idx = {'occurrence':1, 'problem':1, 'treatment':1, 'evidential':1, 'test':1, 'clinical_dept':1}
        for j in range(len(sentences[i])):
            if labels[i][j] == 'O':
                if flag_inlabel:
                    list_dict[i][label.upper()+'_'+str(idx[label])] = ' '.join(sentences[i][idx_start:j])
                    list_out[i].append(label.upper()+'_'+str(idx[label]))
                    idx[label] += 1
                list_out[i].append(sentences[i][j])
                flag_inlabel = 0
            elif 'B' in labels[i][j]:
                label = labels[i][j][2:]
                idx_start = j
                flag_inlabel = 1
                if j == len(labels[i]) - 1:
                    list_dict[i][label.upper()+'_'+str(idx[label])] = sentences[i][j]
                    list_out[i].append(label.upper()+'_'+str(idx[label]))
                    idx[label] += 1
            elif 'I' in labels[i][j] and j == len(labels[i]) - 1:
                list_dict[i][label.upper()+'_'+str(idx[label])] = ' '.join(sentences[i][idx_start:j + 1])
                list_out[i].append(label.upper()+'_'+str(idx[label]))
                idx[label] += 1
                
    list_final = [[] for l in list_out]
    for i in range(len(list_out)):
        idx = {'religion':1, 'insurance':1, 'ethinicity':1, 'numeric': 1}
        for j in range(len(list_out[i])):
            if list_out[i][j].replace('.','',1).isdigit():
                list_dict[i]['NUMERIC'+'_'+str(idx[k])] = list_out[i][j]
                list_final[i].append('NUMERIC'+'_'+str(idx['numeric']))
                idx[k] += 1
            else:
                flag_inkeys = 0
                for k in entity_dict.keys():
                    if sentences[i][j] in entity_dict[k]:
                        #print(idx[k], sentences[i][j])
                        list_dict[i][k.upper()+'_'+str(idx[k])] = sentences[i][j]
                        list_final[i].append(k.upper()+'_'+str(idx[k]))
                        idx[k] += 1
                        flag_inkeys = 1
                    elif j < len(sentences[i]) - 1 and ' '.join(sentences[i][j:j+2]) in entity_dict[k]:
                        list_dict[i][k.upper()+'_'+str(idx[k])] = ' '.join(sentences[i][j:j+2])
                        list_final[i].append(k.upper()+'_'+str(idx[k]))
                        idx[k] += 1
                        flag_inkeys = 1
                    elif j > -1 and ' '.join(sentences[i][j-1:j+1]) in entity_dict[k]:
                        flag_inkeys = 1
                if not flag_inkeys:
                    list_final[i].append(list_out[i][j])

    list_final = [' '.join(l) for l in list_final]                      
    for i in range(len(sentences)):
        print(list_final[i])
        print(list_dict[i])
        print('')
    #return list_final, list_dict