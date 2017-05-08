#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import codecs
import shutil
import warnings
from utils import get_udpipemodel, UDPipe, read_raw_file

"""

Creates CoNLLU training sets with predicted POStags (given by UDpipe)

PYTHONPATH="lysfastparse" python lysfastparse/scripts/gold_to_predicted_conll.py \
GOLDPOS_DIR_TREEBANKS \
DEST_PREDICTEDPOS_DIR_TREEBANKS \
DIR_UDPIPE_MODEL \
PATH_BIN_UDPIPE

PYTHONPATH="lysfastparse" python lysfastparse/scripts/gold_to_predicted_conll.py /data/david.vilares/ud-treebanks-conll2017-D \
/data/david.vilares/ud-treebanks-conll2017-D-pPOS \
/data/david.vilares/UDpipe/udpipe-ud-2.0-conll17-170315/models \
/opt/udpipe-1.1.0-bin/bin-linux64/udpipe
"""

path_dir_gold_treebanks = sys.argv[1]
path_dir_dest_predicted_treebanks = sys.argv[2] 
path_dir_udpipe_models = sys.argv[3]
path_udpipe_bin  = sys.argv[4]


path_gold_treebanks = [(path_dir_gold_treebanks+os.sep+f,f)
                       for f in  os.listdir(path_dir_gold_treebanks)]

for path_gold_treebank, name_treebank in path_gold_treebanks:
    
    print ("Predicting tags for "+path_gold_treebank)   
    
    path_train_conll_treebank = [(path_gold_treebank+os.sep+f,f) for f in os.listdir(path_gold_treebank)
                               if f.endswith("train.conllu")]
    
    path_dev_conll_treebanks = [(path_gold_treebank+os.sep+f,f) for f in os.listdir(path_gold_treebank)
                               if f.endswith("dev.conllu")]
    
    if len(path_train_conll_treebank) != 1:
        warnings.warn("Path "+path_gold_treebank+" contains zero or more than one training data file in conllu format")
    
    if len(path_dev_conll_treebanks) != 1:
        warnings.warn("Path "+path_gold_treebank+" contains zero or more than one development data file in conllu format")

    path_udpipe_model = get_udpipemodel(name_treebank, 
                                              path_dir_udpipe_models)
    
    print ("Using UDpipe model"+path_udpipe_model)
    
    udpipe_model = UDPipe(path_udpipe_model,path_udpipe_bin)   
    
    path_dest_treebank = path_dir_dest_predicted_treebanks+os.sep+name_treebank
        
    if not os.path.exists(path_dest_treebank):
        os.mkdir(path_dest_treebank)
    
    with codecs.open(path_train_conll_treebank[0][0]) as f_train:
        conllu_file = f_train.read()

    tagged_conllu  = udpipe_model.run(conllu_file, options ="--input conllu --tag")

    with codecs.open(path_dest_treebank+os.sep+path_train_conll_treebank[0][1],"w") as f_train_out:
        f_train_out.write(tagged_conllu)

    #Copying development file    
    if len(path_dev_conll_treebanks) == 1:
        path_dev_conll_treebank, name_dev_conll_treebank = path_dev_conll_treebanks[0]
        path_dev_dest_conll_treebank = path_dir_dest_predicted_treebanks+os.sep+name_treebank
        print "Copying development file "+path_dev_conll_treebank+" into "+path_dest_treebank+os.sep+name_dev_conll_treebank
        shutil.copy(path_dev_conll_treebank, path_dest_treebank+os.sep+name_dev_conll_treebank)


            