#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
import json
import os
import pickle
import codecs
import time
import lysfastparse.utils
import lysfastparse.bcovington.utils_bcovington
import lysfastparse.bcovington.covington
import tempfile
import yaml

LTCODE="lcode"
GOLDFILE="goldfile"
OUTFILE="outfile"
PSEGMORFILE="psegmorfile"
RAWFILE="rawfile"
NAME_TREEBANK="name"


R_RAW = "raw"
R_UDPIPE = "udpipe"

YAML_UDPIPE=""

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-c", dest="c", help="Input dataset",metavar="FILE")
    parser.add_argument("-r", dest="r",help="Input run [raw|conllu]", type=str)
    parser.add_argument("-o", dest="o",help="Output directory",metavar="FILE")
    parser.add_argument("-m", dest="m", help="Models directory",metavar="FILE")
    parser.add_argument("-e", dest="e", help="Embeddings directory",metavar="FILE")
    parser.add_argument("--dynet-mem", dest="dynet-mem", help="It is needed to specify this parameter")
    parser.add_argument("--conf", dest="conf")
    
    
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.conf))
    
    with open(args.c+os.sep+"metadata.json") as data_file:    
        metadata_datasets = json.load(data_file)
    
    for metadata in metadata_datasets:
        
        path_model = os.sep.join([args.m,metadata[NAME_TREEBANK],metadata[NAME_TREEBANK]+"-"+metadata[LTCODE]+".model"])
        path_params = os.sep.join([args.m,metadata[NAME_TREEBANK],"params.pickle"])
        
        
        print path_model
        print path_params
        print args.r, type(args.r)
        
        if os.path.exists(os.sep.join([args.o,metadata[OUTFILE]])):
            print os.sep.join([args.o,metadata[OUTFILE]]),"has been already computed"
            continue
        
        
        valid_content = False
        
        if args.r == "conllu" and os.path.exists(path_model):
                 
            with codecs.open(os.sep.join([args.c,metadata[PSEGMORFILE]])) as f:
                 
                f_temp = tempfile.NamedTemporaryFile("w", delete=False)
                f_temp.write(f.read())
                f_temp.close()
                valid_content = True
                 
        elif args.r == "raw" and os.path.exists(path_model):
             
            pipe = lysfastparse.utils.UDPipe(args.udpipe_model, config[YAML_UDPIPE])    
            raw_content = lysfastparse.utils.read_raw_file(args.input)
            conllu = pipe.run(raw_content, options=" --tokenize --tag")
            f_temp = tempfile.NamedTemporaryFile("w", delete=False)
            f_temp.write(conllu)
            f_temp.close()
            valid_content = True
            
    
        if valid_content == True:            
            #TEST PHASE
            with codecs.open(path_params, 'r') as paramsfp:
                aux = pickle.load(paramsfp)
                words, w2i, lemmas, l2i, cpos , pos, feats, rels, stored_opt = aux   
                
            print stored_opt
            print
                
            d = vars(stored_opt)
            
            d["external_embedding"] = None #os.sep.join([args.e,"FB_embeddings","wiki."+metadata[LTCODE]+".vec"])    
            d["pos_external_embedding"] = os.sep.join([args.e,"UD_POS_embeddings",metadata[NAME_TREEBANK]])
            d["feats_external_embedding"] = os.sep.join([args.e,"UD_FEATS_embeddings",metadata[NAME_TREEBANK]])
            d["lemmas_external_embedding"] = None
            
            print d
            print
            stored_opt =Namespace(**d)
            print stored_opt
            print

            print "Running "+path_model
                
            parser = lysfastparse.bcovington.covington.CovingtonBILSTM(words, lemmas, cpos, pos, feats, rels, w2i, l2i, stored_opt,
                                                                           None)
                
            parser.Load(path_model)
                
            with codecs.open(f_temp.name) as f_temp:
                
                lookup_conll_data = lysfastparse.utils.lookup_conll_extra_data(f_temp)
                
            testpath = f_temp.name 
            ts = time.time()
            pred = list(parser.Predict(testpath))
            te = time.time()
            print "Took "+str(te - ts)+" seconds"
            lysfastparse.bcovington.utils_bcovington.write_conll(testpath, pred)
                
            lysfastparse.utils.dump_lookup_extra_into_conll(testpath, lookup_conll_data)
            lysfastparse.utils.transform_to_single_root(testpath)
                
            with codecs.open(os.sep.join([args.o,metadata[OUTFILE]]),"w") as f_out:
                
                with codecs.open(f_temp.name) as f_out_aux:
                    f_out.write(f_out_aux.read())
    
            os.unlink(f_temp.name)
    