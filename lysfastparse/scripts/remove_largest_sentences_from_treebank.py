'''
Created on 18 Apr 2017

@author: david.vilares

python remove_largest_sentences_from_treebank.py \
--input /data/david.vilares/ud-treebanks-conll2017 \
--output /data/david.vilares/ud-treebanks-conll2017-<200 \
--threshold 200
'''
from argparse import ArgumentParser
import os
import codecs
import shutil

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to the directory containing the UDtreebanks",default=None, metavar="FILE")
    parser.add_argument("--output", dest="output",help="Path to the directory that will contain the processed UDtreebanks", metavar="FILE")
    parser.add_argument("--threshold", type=int, dest="threshold", help="Remove sentences larger than threshold")
    
    args = parser.parse_args()
    
    
    path_treebanks = [(args.input+os.sep+d,d) for d in os.listdir(args.input)]
    
    for path_treebank,name_treebank in path_treebanks:
        
        path_dest_treebank = args.output+os.sep+name_treebank
        if not os.path.exists(path_dest_treebank):
            os.mkdir(path_dest_treebank)
        
        path_train_conll_treebank = [(path_treebank+os.sep+f,f) for f in os.listdir(path_treebank)
                               if f.endswith("train.conllu")]
    
        path_dev_conll_treebanks = [(path_treebank+os.sep+f,f) for f in os.listdir(path_treebank)
                               if f.endswith("dev.conllu")]
            
        if len(path_train_conll_treebank) == 1:
            
            with codecs.open(path_train_conll_treebank[0][0]) as f_train:
                sentences = [sentence.split("\n") for sentence in f_train.read().split("\n\n")]
            
            with codecs.open(path_dest_treebank+os.sep+path_train_conll_treebank[0][1],"w") as f_train_out:
                for s in sentences:
                    if len(s) <= args.threshold:
                        f_train_out.write('\n'.join(s))
                        f_train_out.write("\n\n")
                
        if len(path_dev_conll_treebanks) == 1:
            path_dev_conll_treebank, name_dev_conll_treebank = path_dev_conll_treebanks[0]
            path_dev_dest_conll_treebank = args.output+os.sep+name_treebank
            print "Copying development file "+path_dev_conll_treebank+" into "+path_dest_treebank+os.sep+name_dev_conll_treebank
            shutil.copy(path_dev_conll_treebank, path_dest_treebank+os.sep+name_dev_conll_treebank)