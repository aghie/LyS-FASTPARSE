#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
import json
import os
import pickle
import codecs
import time
import lysfastparse.utils
import lysfastparse.bcovington.utils_bcovington
import tempfile
import yaml
import subprocess
import sys

LCODE="lcode"
TCODE="tcode"
GOLDFILE="goldfile"
OUTFILE="outfile"
PSEGMORFILE="psegmorfile"
RAWFILE="rawfile"
NAME_TREEBANK="name"


R_RAW = "raw"
R_UDPIPE = "udpipe"

YAML_UDPIPE="udpipe"


def get_models_dict(path_models):
    d = {}
    files = [(path_models+os.sep+f,f) for f in os.listdir(path_models)]
    for path,name in files:
        name_split =name.split(".")
        l,t = name_split[0],name_split[1]
        
        if l not in d: d[l] = {}
        if t not in d[l]: d[l][t] = {"model":None,
                                     "params":None}
        
        if name.endswith(".model"):
            d[l][t]["model"] = path 
        if name.endswith(".pickle"):
            d[l][t]["params"] = path 
    
    return d


def select_model(lcode, tcode, dict_models):
    
    print dict_models
    print lcode, tcode
    
    try:
        #If we know the lang and treebank code
        return dict_models[lcode][tcode]["model"],dict_models[lcode][tcode]["params"]  
    except KeyError:
        try:
            #If we know the lang but not the treebank code
            return  dict_models[lcode]["0"]["model"],dict_models[lcode]["0"]["params"]  
        except KeyError:
            #We do not know the lang neither the treebank code
            if "en" in dict_models:
                return dict_models["en"]["0"]["model"],dict_models["en"]["0"]["params"]
            else:
                return None,None


def get_udpipe_models(path_models):
    
    d = {}
    files = [(path_models+os.sep+f,f) for f in os.listdir(path_models)]
    for path,name in files:
        name_udpipe_model =name.split("-ud-")[0].lower()
        
        d[name_udpipe_model] = path 

    return d
    
def select_udpipe_model(name_treebank,dict_udpipe_models):
    
    try:
        return dict_udpipe_models[name_treebank]
    except KeyError:
        try:
            return dict_udpipe_models[name_treebank.split("-")[0]]
        except KeyError:
            return dict_udpipe_models["english"]

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-c", dest="c", help="Input dataset",metavar="FILE")
    parser.add_argument("-r", dest="r",help="Input run [raw|conllu]", type=str)
    parser.add_argument("-o", dest="o",help="Output directory",metavar="FILE")
    parser.add_argument("-m", dest="m", help="Models directory",metavar="FILE")
    parser.add_argument("-e", dest="e", help="Embeddings directory",metavar="FILE")
    parser.add_argument("-um", dest="um",help="UDpipe models direcotry", metavar="FILE")
    parser.add_argument("--dynet-mem", dest="dynet_mem", help="It is needed to specify this parameter")
    parser.add_argument("-conf", dest="conf")
    
    
    args = parser.parse_args()
    
    print "args", args
    
    config = yaml.safe_load(open(args.conf))
    
    with open(args.c+os.sep+"metadata.json") as data_file:    
        metadata_datasets = json.load(data_file)
    
    dict_models = get_models_dict(args.m)

    for metadata in metadata_datasets:
        
        path_model, path_params = select_model(metadata[LCODE], metadata[TCODE], dict_models)
        if path_model is None: continue

        path_udpipe_bin = "none"
        path_udpipe_model = "none"
        name_extrn_emb = path_model.rsplit("/",1)[1].split(".")[2]
        
        print "Processing file", metadata[PSEGMORFILE]
        print "Model:",path_model
        print "Params:",path_params
        print "POS/FEATS embeddings: ",name_extrn_emb
                 
        path_pos_embeddings = os.sep.join([args.e,"UD_POS_embeddings",name_extrn_emb])
        path_feats_embeddings = os.sep.join([args.e,"UD_FEATS_embeddings",name_extrn_emb])
        path_embeddings = os.sep.join([args.e,"word-embeddings-conll17",metadata[LCODE]+".vectors"])
        path_output = os.sep.join([args.o,metadata[OUTFILE]])
        if args.r == "conllu":
            print "Parsing the segmor output from UDPipe"
            path_input = os.sep.join([args.c,metadata[PSEGMORFILE]])
        elif args.r == "raw":
            print "Using the raw file with..."
            dict_udpipe_models = get_udpipe_models(args.um)
            path_input = os.sep.join([args.c,metadata[RAWFILE]])
            path_udpipe_bin = config[YAML_UDPIPE]
            path_udpipe_model = select_udpipe_model(name_extrn_emb.replace("UD_","").lower(),dict_udpipe_models)
            print "path_udpipe_model", path_udpipe_model
        else:
            raise NotImplementedError
        

        if os.path.exists(path_output):
            print path_output,"has been previously computed"
        elif not os.path.exists(path_model):
            print path_output,"there is no", path_model," model"
        else:
            command = " ".join(["python run_model.py", "-p",path_params,"-m",path_model, 
                                "-o",path_output, "-epe", path_pos_embeddings, "-efe", path_feats_embeddings,
                                "-ewe", path_embeddings,
                                "-r",args.r, "-i",path_input,
                                "--dynet-mem", args.dynet_mem,
                                "-udpipe_bin", path_udpipe_bin,
                                "-udpipe_model", path_udpipe_model])

            
            os.system(command)
            

    
