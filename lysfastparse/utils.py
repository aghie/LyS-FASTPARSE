from ufal.udpipe import *
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
        
        command = self.udpipe+' '+self.path_model+' '+f_temp.name+' '+options
        p = subprocess.Popen([command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True);           
        output, err = p.communicate()

        if err is not None:
            warnings.warm("Something unexpected occurred when running: "+command)
        
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
        

