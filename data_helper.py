import csv
import re
import numpy as np
import torch
from torch.autograd import Variable
import torchtext.datasets as datasets
import os
import random
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords


cachedStopWords = stopwords.words("english")


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()

def line_to_words(line):
    clean_line = clean_str_sst(line)
    # print(f"clean_line:{clean_line}")
    Porter_Stemmer=PorterStemmer()
    words = ' '.join([Porter_Stemmer.stem(word) for word in clean_line.split() if word not in cachedStopWords])
    # print(f"words:{words}")
    return words

def convert_to_tsv(path):
    c=0
    d=0
    neg_test,pos_test,neg_train,pos_train=[],[],[],[]
    for items in os.scandir(path+'/data/test/neg'):
        if items.is_file():
            fn_test=open(items,'r').read()
            neg_test.append(fn_test)
    for items in os.scandir(path+'/data/test/pos'):
        if items.is_file():
            fp_test=open(items,'r').read()
            pos_test.append(fp_test)
    for items in os.scandir(path+'/data/train/neg'):
        if items.is_file():
            fn_train=open(items,'r').read()
            neg_train.append(fn_train)
    for items in os.scandir(path+'/data/train/pos'):
        if items.is_file():
            fp_train=open(items,'r').read()
            pos_train.append(fp_train)
    test,train,dev=[],[],[]
    for line in neg_test:
        words=line_to_words(line)
        sent = words + '\t' + '0'
        test.append(sent)
    for line in pos_test:
        words=line_to_words(line)
        sent = words + '\t' + '1'
        test.append(sent)
    for line in neg_train:
        words=line_to_words(line)
        sent = words + '\t' + '0'
        train.append(sent)
    for line in pos_train:
        words=line_to_words(line)
        sent = words + '\t' + '1'
        train.append(sent)
    random.shuffle(test)
    random.shuffle(train)
    test,dev=test[:len(test)//2],test[len(test)//2:]
    train.sort(key=lambda x: len(x))

    train_out, dev_out, test_out = open(path+'/FOR_USE/train.tsv', 'w'),\
                                   open(path+'/FOR_USE/dev.tsv', 'w'),\
                                   open(path+'/FOR_USE/test.tsv', 'w')
                        
    train_out.write('\n'.join(train))
    dev_out.write('\n'.join(dev))
    test_out.write('\n'.join(test))
    return train,dev,test


if __name__ == '__main__':
    train, dev, test = convert_to_tsv('/Users/hutianxing/vscodeprojects/nlp_homework_3')
    #print(train)
