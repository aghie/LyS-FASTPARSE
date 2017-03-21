from ufal.udpipe import *
import codecs
import sys

def read_raw_file(path):
    with codecs.open(path) as f:
        return f.read().replace('\n',' ')


class UDPipe(object):
    
    def __init__(self,path_model):
        self.model = Model.load(path_model)

        
    def run(self, text):
        pipeline = Pipeline(self.model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")    
        error = ProcessingError()
        
        print "Processing %s" % text
        processed = pipeline.process(text, error)
        if error.occurred():
            sys.stderr.write("An error occurred when running run_udpipe: ")
            sys.stderr.write(error.message)
            sys.stderr.write("\n")
            sys.exit(1)
        sys.stdout.write(processed)
        return processed

