import codecs
import sys
import os
import subprocess
import tempfile
import warnings

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
    
"""
It tries to solve some tokenizing error that were often observed in UDpipe:
- ")" stays appended to the word in many cases
-
"""
def custom_tokenizing(text):
    raise NotImplementedError
        

def get_udpipemodel(name_treebank, path_models):
    
    udpipe_model = [path_models+os.sep+f for f in os.listdir(path_models) 
     if f.startswith(name_treebank.replace("UD_","").lower()+"-ud")]
    
    if len(udpipe_model) !=1:
        raise ValueError("More than 1 UDpipe model for the specified treebank was found. Cannot decide which one to take")
    
    return udpipe_model[0]



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
            
    
def get_OOV_words_from_conll(word2vec_model, path_conllus):
    
    ovvs = set([])
    for path_conllu in path_conllus:
        with codecs.open(path_conllu) as f_conllu:
            f_conllu.read()

    


def get_facebook_embeddings(name_treebank, path_fb_embeddings):
    
    #TODO add mapping for all FB embeddings
    d = {"Arabic":"wiki.ar.vec",
         "Basque":"wiki.eu.vec",
         "Bulgarian":"wiki.bg.vec",
         "Catalan":"wiki.ca.vec",
         "Chinese":"wiki.zh.vec",
         "Croatian":"wiki.hr.vec",
         "Czech":"wiki.cs.vec",
         "Danish":"wiki.da.vec"}
    
    try:
        return path_fb_embeddings+os.sep+d[name_treebank.replace("UD_","")]
    except KeyError:
        return None
    
