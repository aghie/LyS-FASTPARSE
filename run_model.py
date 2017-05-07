
from argparse import ArgumentParser, Namespace
import codecs
import sys
import pickle
import os
import time
import lysfastparse.utils
import lysfastparse.bcovington.utils_bcovington
import tempfile
import yaml
import subprocess
import lysfastparse.bcovington.covington


parser = ArgumentParser()
parser.add_argument("-p", dest="p",metavar="FILE")
parser.add_argument("-m", dest="m",metavar="FILE")
parser.add_argument("-o", dest="o",metavar="FILE")
parser.add_argument("-epe", dest="epe",metavar="FILE")
parser.add_argument("-efe",dest="efe",metavar="FILE")
parser.add_argument("-ewe",dest="ewe", metavar="FILE")
parser.add_argument("-r", dest="r",help="Input run [raw|conllu]", type=str)
parser.add_argument("-i", dest="i",metavar="FILE")
parser.add_argument("--dynet-mem", dest="dynet_mem", help="It is needed to specify this parameter")
parser.add_argument("--udpipe_bin", dest="udpipe_bin",metavar="FILE")
parser.add_argument("--udpipe_model", dest="udpipe_model",metavar="FILE")
    
args = parser.parse_args()

print "args (run_model.py)",args

path_params = args.p
path_model = args.m
path_outfile = args.o
path_embeddings = args.ewe
path_pos_embeddings = args.epe
path_feats_embeddings = args.efe
type_text = args.r
path_input = args.i


valid_content = False
    
if type_text == "conllu" and os.path.exists(path_model):
             
    with codecs.open(path_input) as f:
             
        f_temp = tempfile.NamedTemporaryFile("w", delete=False)
        f_temp.write(f.read())
        f_temp.close()
        valid_content = True
             
elif type_text == "raw" and os.path.exists(path_model):
         
    pipe = lysfastparse.utils.UDPipe(args.udpipe_model, args.udpipe_bin) #config[YAML_UDPIPE])    
    raw_content = lysfastparse.utils.read_raw_file(path_input)
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
                    
                    
    d = vars(stored_opt)
    
    print d
                
    d["external_embedding"] = None if d["external_embedding"] =="None" else path_embeddings #os.sep.join([args.e,"FB_embeddings","wiki."+metadata[LTCODE]+".vec"])    
    d["pos_external_embedding"] = path_pos_embeddings #os.sep.join([args.e,"UD_POS_embeddings",metadata[NAME_TREEBANK]])
    d["feats_external_embedding"] = path_feats_embeddings #os.sep.join([args.e,"UD_FEATS_embeddings",metadata[NAME_TREEBANK]])
    d["lemmas_external_embedding"] = None
             
    
    print "pos_external_embeddings", d["pos_external_embedding"]
    print "feats_external_embeddings", d["feats_external_embedding"]  
    print "external_embedding", d["external_embedding"]
       
#     print d
#     print
    stored_opt =Namespace(**d)
    print "Running model with this configuration", stored_opt
#     print
#     
#     print "Running "+path_model
                   
                    
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
                        
        with codecs.open(path_outfile,"w") as f_out:
                        
            with codecs.open(f_temp.name) as f_out_aux:
                f_out.write(f_out_aux.read())
            
    os.unlink(f_temp.name)