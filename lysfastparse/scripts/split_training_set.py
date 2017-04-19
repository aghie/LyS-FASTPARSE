'''
Created on 19 Apr 2017

@author: david.vilares

Created a dev set from the original training set for those treebanks that did not have one. 
The original training set is overwritten by the new one.

python split_training_set.py \
--input /data/david.vilares/ud-treebanks-conll2017-alldev
'''

from argparse import ArgumentParser
import os
import codecs
import shutil
import random


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to a directory containing the CoNLLU2017 treebanks",default=None, metavar="FILE")
    parser.add_argument("--split_to_dev", dest="split_to_dev", type=int, default=20,help="Integer between 1 and 99 indicating the percentage of the training set that will be considered as a dev set instead")
    
    args = parser.parse_args()
    
    if args.split_to_dev < 1 or args.split_to_dev > 99:
        raise ValueError("Cannot perform split with value"+str(args.split_to_dev))
    
    
    path_treebanks = [(args.input+os.sep+d,d) for d in os.listdir(args.input)]
    
    for path_treebank, name_treebank in path_treebanks:
    
        path_train_conll = [(path_treebank+os.sep+f,f) for f in os.listdir(path_treebank)
                            if f.endswith('train.conllu')]
        
        path_dev_conll = [(path_treebank+os.sep+f,f) for f in os.listdir(path_treebank)
                            if f.endswith('dev.conllu')]
        
        
        if len(path_dev_conll) ==0:
            
            print ("CREATING DEV SET FOR "+name_treebank)
            path_dev_set = path_treebank+os.sep+path_train_conll[0][1].split("-")[0]+"-ud-dev.conllu"
            path_dev_set_txt = path_treebank+os.sep+path_train_conll[0][1].split("-")[0]+"-ud-dev.txt"
            
            f_dev_set = codecs.open(path_dev_set,"w")
            f_dev_set_txt = codecs.open(path_dev_set_txt,"w")
            
            with codecs.open(path_train_conll[0][0]) as f_train:
                sentences = f_train.read().split("\n\n")
            
            #We now proceed to overwrite the original training set
            #We also create the corresponding txt file
            f_train_set = codecs.open(path_train_conll[0][0],"w")
            f_train_set_txt =  codecs.open(path_train_conll[0][0].replace(".conllu",".txt"),"w")
            
            for i,s in enumerate(sentences):
                    
                if i % 100 < args.split_to_dev:
                    f_dev_set.write(s+"\n\n")
                else:
                    f_train_set.write(s+"\n\n")
                
            f_dev_set.close()
            f_train_set.close()
            
        else:
            print (name_treebank+" already contains a development set")
        
        
        