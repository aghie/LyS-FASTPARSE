#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
import os
import subprocess
import tempfile
import warnings
import subprocess
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.wrappers.fasttext import FastTextKeyedVectors, FastText
import sys 


DUMMY_ROOT = 0
UD_CTAG_VERB = "VERB"
UD_HEAD_COLUMN = 6
UD_CTAG_COLUMN = 3
UD_ID_COLUMN = 0

"""
A simple wrapper for Udpipe
"""
class UDPipe(object):
    
    def __init__(self,path_model, path_udpipe):
        self.path_model = path_model
        self.udpipe = path_udpipe
    

    def run(self,text,options=' --tokenize --tag '):
        
        f_temp = tempfile.NamedTemporaryFile("w", delete=False)
        f_temp.write(text)
        f_temp.close()

        command = self.udpipe+' '+options+' '+self.path_model+' '+f_temp.name
        p = subprocess.Popen([command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True);           
        output, err = p.communicate()

        if err is not None:
            warnings.warn("Something unexpected occurred when running: "+command)
        
        return output
         
         
def read_raw_file(path):
    with codecs.open(path) as f:
        return f.read().replace('\n',' ')
            

def get_udpipemodel(name_treebank, path_models):
    
    udpipe_model = [path_models+os.sep+f for f in os.listdir(path_models) 
     if f.startswith(name_treebank.replace("UD_","").lower()+"-ud")]
    
    if len(udpipe_model) !=1:
        raise ValueError("More than 1 UDpipe model for the specified treebank was found. Cannot decide which one to take")
    
    return udpipe_model[0]


"""
Looks for multiword expressions in the CoNLL file and creates a lookup table that
allows to reconstruct then the output
"""
def lookup_conll_extra_data(fh):
    
    lookup = {}
    sentence_id = 0
    lookup[sentence_id] = {}
    id_insert_before = 1
    
    for line in fh:
        
        if line.startswith('#'): continue
        tok = line.strip().split('\t')

        if not tok or tok == ['']: #If it is empty line
            sentence_id+=1
            id_insert_before = 1
            lookup[sentence_id] = {}
        else:
            if "." in tok[0] or "-" in tok[0]:
                lookup[sentence_id][id_insert_before] = line
            else:
                id_insert_before+=1
 
    return lookup
            
"""
dumps the content of the lookup table extracted by lookup_conll_extra_data
into a output conll_path
"""
def dump_lookup_extra_into_conll(conll_path,lookup):
    
    sentence_id = 0
    word_id = 1

    with codecs.open(conll_path) as f_conll:
        lines = f_conll.readlines()

    #DUMPING the content of the file
    f_conll = codecs.open(conll_path,"w")
    
    for line in lines:
        
        tok = line.strip().split('\t')
        if tok == ['']: #If it is empty line
            sentence_id+=1
            word_id = 1
        else:
            if sentence_id in lookup: 
                if word_id in lookup[sentence_id]:
                    f_conll.write(lookup[sentence_id][word_id])
            word_id+=1
        f_conll.write(line)
        
    f_conll.close()


def get_rooted(conll_str):
    """
    Returns a list of [id,ctag,head] of the nodes rooted to 0
    """
    rooted_elements = []
    
    lines = conll_str.split('\n')
    for l in lines:
        ls = l.split('\t')
        try:
            identifier,tag,head = int(ls[UD_ID_COLUMN]),ls[UD_CTAG_COLUMN],int(ls[UD_HEAD_COLUMN])
            if head == DUMMY_ROOT:
                rooted_elements.append((identifier,tag,head))      
        except ValueError:
            pass   
    return rooted_elements
    

def get_new_single_root(lmultiple_rooted):
    """
    Returns the ID of the first VERB rooted to 0 or the leftmost rooted
    element otherwise
    """
    for e in lmultiple_rooted:
        if e[2] == DUMMY_ROOT and e[1] == UD_CTAG_VERB:
                return e[0]     
    return lmultiple_rooted[0][0]
            
"""
"""
def transform_to_single_root(conll_path):
    
    with codecs.open(conll_path) as f_conll:
        sentences = f_conll.read().split('\n\n')
    
    with codecs.open(conll_path,"w") as f_conll:
        
        i=0
        for s in sentences:
            if s == "": continue
            rooted = get_rooted(s)
            if len(rooted) > 1:
                frv = get_new_single_root(rooted)
                for l in s.split('\n'):
                    ls = l.strip().split('\t')   
                    
                    if ls != [''] and not l.startswith("#"): #If it is empty line
                        if ls[UD_HEAD_COLUMN] != "_" and int(ls[UD_HEAD_COLUMN]) == DUMMY_ROOT and int(ls[UD_ID_COLUMN]) != frv:
                            ls[UD_HEAD_COLUMN] = str(frv)
                        
                    f_conll.write('\t'.join(ls)+"\n")
            else:
                f_conll.write(s+"\n") 
            f_conll.write('\n')
            i+=1
        
    
def get_OOV_words_from_conll(path_fasttext,
                             path_w2v_FBbin, path_w2v_FBtxt, words):
    
    ovvs = set([])


    #model = FastText.load_fasttext_format(path_w2v_FBtxt.replace(".vec",""))
    
    with codecs.open(path_w2v_FBtxt) as f:
        size = f.readline().split()[1]
        
    model =  gensim.models.KeyedVectors.load_word2vec_format(path_w2v_FBtxt)
    
#     print len(words)
#     for word in words:
#         if word not in model:
#             print word, word.replace(" ","")
        

    f_oov = tempfile.NamedTemporaryFile("w", delete=False)
    #To unify compound words
    oovs = set([word.replace(" ","_") for word in words 
                if  word not in model])

    f_oov.write("\n".join(oovs))
    f_oov.close()

    command = path_fasttext+" print-vectors "+path_w2v_FBbin+" < "+f_oov.name
    p = subprocess.Popen([command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)           
    #os.system(command)
    #exit()
    output, err = p.communicate()

    output = output.decode("utf-8")
    
    f_oov_embeddings = tempfile.NamedTemporaryFile("w", delete=False) 
    
    f_oov_embeddings.write(str(len(oovs))+" "+str(size)+"\n")
    f_oov_embeddings.write(output)
    f_oov_embeddings.close()
    
    os.unlink(f_oov.name)
    
    return f_oov_embeddings.name



    
