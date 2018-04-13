import conllu
import os
import pandas as pd
import numpy as np
import re

from os import listdir
from os.path import isfile, isdir, join
from xml.etree import ElementTree

def conllu_meta_parse_line(line):
    id_m = re.match(r"#ID=\/[0-9_]+\/([\w]+).xml", line)
    sent_m = re.match(r"#SENT=([^\n]+)", line)
    if id_m:
        return id_m.group(1)
    if sent_m:
        return sent_m.group(1)

def conllu_meta_parse(text):
    meta_infos = [
        [
            conllu_meta_parse_line(line)
            for line in sentence.split("\n")
            if line and line.strip().startswith("#")
        ]
        for sentence in text.split("\n\n")
        if sentence
    ]
    return [{"doc_id": meta_info[0], "sent": meta_info[1]} for meta_info in meta_infos]

def load_raw_conllu(load_train=True, load_dev=True, load_test=True):
    print("Load raw conllu dataset (load_train={}, load_dev={}, load_test={})".format(load_train, load_dev, load_test))
    filepaths = []
    if load_train:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-train.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-train.conllu",
        ]
    if load_dev:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-dev.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-dev.conllu", 
        ]
    if load_test:
        filepaths += [
            "dataset/UD_English-ESL/data/en_esl-ud-test.conllu", 
            "dataset/UD_English-ESL/data/corrected/en_cesl-ud-test.conllu",
        ]
    data_list = []
    meta_list = []
    for path in filepaths:
        print("Processing {}".format(path))
        with open(path, "r") as f:
            raw_data = f.read()
        data_list.append(conllu.parse(raw_data))
        meta_list.append(conllu_meta_parse(raw_data))
        
    return meta_list, data_list

def load_post_metadata():
    filepath = "dataset/UD_English-ESL/fce-released-dataset/dataset/"
    subpaths = [f for f in listdir(filepath) if isdir(join(filepath, f))]
    stats_raw = []
    for subpath in subpaths:
        subpath = join(filepath, subpath)
        files = [f for f in listdir(subpath) if isfile(join(subpath, f))]
        for file in files:
            l_id = file.split('.')[0]
            file = join(subpath, file)
            with open(file, 'rt') as f:
                tree = ElementTree.parse(f)
            for learner in tree.iter('learner'):
                l_native_lan = None
                l_age = None
                l_score = None
                for candidate in learner.iter('candidate'):
                    for personnel in candidate.iter('personnel'):
                        for language in personnel.iter('language'):
                            l_native_lan = language.text
                        for age in personnel.iter('age'):
                            l_age = age.text
                for score in learner.iter('score'):
                    l_score = float(score.text)
                stats_raw.append([l_id, l_native_lan, l_age, l_score])
    stats_df = pd.DataFrame(stats_raw, columns=['doc_id', 'native_language', 'age_range', 'score'])
    return stats_df

def load_data(load_train=True, load_dev=True, load_test=True):
    print("Load data (load_train={}, load_dev={}, load_test={})".format(load_train, load_dev, load_test))
    raw_meta_list, raw_data_list = load_raw_conllu(load_train, load_dev, load_test)
    print("Build metadata")
    meta_list = []
    for meta in raw_meta_list:
        cols = list(meta[0].keys())
        metas = []
        for i in range(len(meta)):
            values = list(meta[i].values())
            metas.append(values)
        meta_list.append(pd.DataFrame(metas, columns=cols))
        meta_list[-1].insert(0, 'id', range(1, len(meta_list[-1])+1))
    print("Build sentences")
    data_list = []
    for idx, data in enumerate(raw_data_list):
        meta = meta_list[idx]
        sentence_dfs = []
        cols = list(data[0][0].keys()) + ["meta_id"]
        for i in range(len(data)):
            words = []
            for j in range(len(data[i])):
                words.append(list(data[i][j].values()) + [meta["id"][i]])
            sentence_dfs.append(pd.DataFrame(words, columns=cols).set_index('id'))
        data_list.append(sentence_dfs)
    post_df = load_post_metadata()
    for i in range(len(meta_list)):
        meta_list[i] = meta_list[i].set_index('doc_id').join(post_df.set_index('doc_id'))
        meta_list[i] = meta_list[i].reset_index().set_index('id').sort_index()
    print("=================")
    return meta_list, data_list

def dump_preprocessed_data(name, meta, data, format="csv"):
    print("Dump {} to preprocessed/{}".format(name, name))
    directory = "preprocessed/{}".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)  
    if format == "csv":
        meta.to_csv("{}/meta.csv".format(directory))
    else:
        meta.to_json("{}/meta.json".format(directory))
    for d in data:
        if format == "csv":
            d.to_csv("{}/{}.csv".format(directory, d["meta_id"].unique()[0]))
        else:
            d.to_json("{}/{}.json".format(directory, d["meta_id"].unique()[0]))

def main():
    meta_list, data_list = load_data(load_train=True, load_dev=True, load_test=True)

    train_meta, train_meta_corrected, \
    dev_meta, dev_meta_corrected, \
    test_meta, test_meta_corrected = meta_list
    
    train_data, train_data_corrected, \
    dev_data, dev_data_corrected, \
    test_data, test_data_corrected = data_list
    
    f = "csv" # or "json"
    dump_preprocessed_data("train", train_meta, train_data, format=f)
    dump_preprocessed_data("train_corrected", train_meta_corrected, train_data_corrected, format=f)
    dump_preprocessed_data("dev", dev_meta, dev_data, format=f)
    dump_preprocessed_data("dev_corrected", dev_meta_corrected, dev_data_corrected, format=f)
    dump_preprocessed_data("test", test_meta, test_data, format=f)
    dump_preprocessed_data("test_corrected", test_meta_corrected, test_data_corrected, format=f)

if __name__ == "__main__":
    main()