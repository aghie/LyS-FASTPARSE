from ufal.udpipe import *
import codecs
import sys
import os
import subprocess
import tempfile
import warnings

def read_raw_file(path):
    with codecs.open(path) as f:
        return f.read().replace('\n',' ')


class UDPipe(object):
    
    def __init__(self,path_model, path_udpipe):
        self.path_model = path_model
        #self.model = Model.load(path_model)
        self.udpipe = path_udpipe


    def run(self,text,options=' --tokenize --tag '):
        #codecs.open("/tmp/proof.conllu","w")
        f_temp = tempfile.NamedTemporaryFile("w", delete=False)
        f_temp.write(text)
        f_temp.close()
        command = self.udpipe+' '+self.path_model+' '+f_temp.name+' '+options
#         print command
#         os.system(command)
        p = subprocess.Popen([command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True);           
        #encode needed for non ascii texts (e.g. Spanish texts)
        output, err = p.communicate()

         
        if err is not None:
            warnings.warm("Something unexpected occurred when running: "+command)
        
        return output
         
        
# I do not see the option in the python UDpipe to allow only tokenizing and tagg
#     def run(self, text):
#         pipeline = Pipeline(self.model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")    
#         error = ProcessingError()
#         
#         print "Processing %s" % text
#         processed = pipeline.process(text, error)
#         if error.occurred():
#             sys.stderr.write("An error occurred when running run_udpipe: ")
#             sys.stderr.write(error.message)
#             sys.stderr.write("\n")
#             sys.exit(1)
#         sys.stdout.write(processed)
#         return processed

