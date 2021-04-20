import os
import re
import pandas as pd
import glob


def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files

files = glob.glob("CliNER/data/test_con_file_n2c2/*.con")

PATH = 'CliNER/data/test_con_file_n2c2'
con_files = list_files(PATH)
print(con_files)
print(files)

#df_name_list = []
for e in files:
    #file= '/Users/isabelmetzger/PycharmProjects/NaturalLanguageInterface/CliNER/data/predictions_for_loop/100.con'
    infile = open(e, 'r')
    dictionary_pt = {"NER": [],
                "concept_text": [],
                 "concept_tag": [], "beginning": [], "end": []}
    print("Creating csv file for:", e)
    #df_name = os.path.splitext(e)[0] +
    #df_name = str(e) + ".Dataframe"
    #print(df_name)
    list_sentences = []
    for line in infile:
        list_sentences.append(line)

    t_l = []
    c_l = []
    i = 0
    numbers = []
    n_l_start = []
    for i in range(len(list_sentences)):
        dictionary_pt["NER"].append(list_sentences[i])
        c = re.findall(r'c\=\"(.+?)\"', str(list_sentences[i]))
        print(c)
        c_l.append(c)
        dictionary_pt["concept_text"] = [item for sublist in c_l for item in sublist]
        t = re.findall(r't=\"(.+?)\"', str(list_sentences[i]))
        t_l.append(t)
        dictionary_pt["concept_tag"] = [item for sublist in t_l for item in sublist]
        numbers.append(re.findall(r"\d+\:\d*", str(list_sentences[i])))

    for list_i in numbers:
        dictionary_pt["beginning"].append(list_i[0])
        dictionary_pt["end"].append(list_i[0])
    print("head of csv file for", e)
    #print(pd.DataFrame(dictionary_pt).head(2))
    df_patient = pd.DataFrame(dictionary_pt)
    print(df_patient.head())
    #df_patient.to_csv()
    #print("File " + "patient" + str(os.path.splitext(e)[0]) + ".csv" + " created")

    #df_name_list.append(df_name)
    #
    # #
    # # # file = '/Users/isabelmetzger/PycharmProjects/NaturalLanguageInterface/CliNER/data/predictions_for_loop/185.con'
    # infile = open(e, 'r')
    # dictionary_pt = {"NER": [],
    #               "concept_text": [],
    #                  "concept_tag": [],# "locations": []
    #                  "beginning": [], "end": []
    #
    #                  #     "line_idx_start": [],
    #                  #     "line_idx_end": [],
    #                  #     "word_idx_start": [],
    #                  #     "word_idx_end": [], "text_location": []
    #                  }
    #
    # list_sentences = []
    # for line in infile:
    #     list_sentences.append(line)
    #
    #     t_l = []
    #     c_l = []
    #     i = 0
    #
    #     numbers = []
    #     for i in range(len(list_sentences)):
    #         dictionary_pt["NER"].append(list_sentences[i])
    #         c = re.findall(r'c\=\"(.+?)\"', str(list_sentences[i]))
    #         #print(c)
    #         c_l.append(c)
    #         dictionary_pt["concept_text"] = [item for sublist in c_l for item in sublist]
    #         t = re.findall(r't=\"(.+?)\"', str(list_sentences[i]))
    #         t_l.append(t)
    #         dictionary_pt["concept_tag"] = [item for sublist in t_l for item in sublist]
    #         numbers.append(re.findall(r"\d+\:\d*", str(list_sentences[i])))
    #         #dictionary_pt["locations"] = [item for sublist in c_l for item in sublist]
    #     for list_i in numbers:
    #         dictionary_pt["beginning"].append(list_i[0])
    #         dictionary_pt["end"].append(list_i[0])
    #
    #
    # print(len(dictionary_pt))
    # #print(glance(dictionary_pt))
    #
    # df_patient = pd.DataFrame(dictionary_pt)
    # df_patient.to_csv("Patient" + str(os.path.splitext(e)[0]) + ".csv")#, sep=',')
    # print("File " + "patient" + str(os.path.splitext(e)[0]) + ".csv" + " created")
    #
    #
    #
    # # w_lines = []
    # # list_groups = []
    # # t_g1 = 0
    # # t_g2 = 0
    # # t_g4 = 0
    # # with open(e) as f:
    # #     for line in f.readlines():
    # #         if not line.strip():
    # #             continue
    # #         concept_regex = 'c="(.*)" (\d+):(\d+) (\d+):(\d+)\|\|type="(\w*)"'
    # #         match = re.search(concept_regex, line.strip())
    # #         print(match.groups())
    # #     #     if match:
    # #     #         groups = match.groups()
    # #     #         list_groups.append(groups)
    # #     #
    # #     # # sort annotations according to line indent
    # #     # list_groups = sorted(list_groups, key=lambda x: int(x[1]))
    # #     #
    #     # # delete duplicate annotations
    #     # for groups in list_groups:
    #     #
    #     #     if not (int(groups[1]) == t_g1 and t_g2 <= int(groups[2]) and int(groups[2]) <= t_g4):
    #     #         t_g1 = int(groups[1])
    #     #         t_g2 = int(groups[2])
    #     #         t_g4 = int(groups[4])
    #     #         if groups[5] != '':
    #     #             w_lines.append(
    #     #                 ',' + groups[0] + ', ' + groups[1] + ':' + groups[2] + ' ' + groups[3] + ':' + groups[
    #     #                     4] + '||t="' + groups[5].lower() + '"')
    #     #         if os.path.splitext(e)[0] == 'test_con\88':
    #     #           print(w_lines)
    #     # #create .con files
    #     # f = open(os.path.splitext(e)[0])
    #     # f.write('\n'.join(w_lines))
    #     # '\n'.join(w_lines)
    #     # f.close()
