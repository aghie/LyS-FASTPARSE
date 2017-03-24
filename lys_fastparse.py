from argparse import ArgumentParser
from lysfastparse import utils
from bcovington import covington
import bcovington.utils
import os
import pickle
import time
import tempfile
import yaml
"""
Example of execution


python lys_fastparse.py \
--dynet-seed 123456789 --dynet-mem 4000 \
--input_type raw \
--outdir /tmp/ \
--train /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Basque/eu-ud-train.conllu \
--dev /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Basque/eu-ud-dev.conllu \
--test /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Basque/eu-ud-test.conllu \
--epochs 1 \
--lstmdims 125 \
--lstmlayers 2 \
--bibi-lstm \
--k1 3 \
--k2r 0 \
--k2l 0 \
--usehead \
--userl \
--predict \
--input  /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Spanish/es-ud-train-proof.txt \
--model /home/david.vilares/Escritorio/Papers/bist-covington/UD_Basque-optimizer:adam-lstmdims:125-extrn=True-activation:tanh-pembeddings:25-kb:1-k2r:0-wembeddings:125-k1:2-k2l:0-rembeddings:25/barchybrid.model10 \
--params /home/david.vilares/Escritorio/Papers/bist-covington/UD_Basque-optimizer:adam-lstmdims:125-extrn=True-activation:tanh-pembeddings:25-kb:1-k2r:0-wembeddings:125-k1:2-k2l:0-rembeddings:25/params.pickle \

"""

#YAML ATTRIBUTES
YAML_UDPIPE = "udpipe"

#INPUT TYPES
INPUT_RAW = "raw"
INPUT_CONLLU = "conllu"

#AVAILABLE PIPELINES
PIPELINE_UDPIPE = "UDpipe"

#TODO: Polish this
UDPIPE_MODEL = "/data/UDpipe/models/gl_udv2"

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to the input file",default=None)
    parser.add_argument("--input_type", dest="input_type",help="Style of the input file [raw|conllu]")
    parser.add_argument("--pipe", dest="pipe",default="UDpipe",help="Framework used to do the pipeline. Only \"UDpipe\" supported")
    
    parser.add_argument("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/PTB_SD_3_3_0/train.conll")
    parser.add_argument("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/PTB_SD_3_3_0/dev.conll")
    parser.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/PTB_SD_3_3_0/test.conll")
    parser.add_argument("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_argument("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_argument("--extrn_cpos", dest="cpos_external_embedding",help="CPoStag external embeddings", metavar="FILE")
    parser.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_argument("--wembedding", type=int, dest="wembedding_dims", default=100)
    parser.add_argument("--pembedding", type=int, dest="pembedding_dims", default=25)
    parser.add_argument("--rembedding", type=int, dest="rembedding_dims", default=25)
    parser.add_argument("--epochs", type=int, dest="epochs", default=30)
    parser.add_argument("--hidden", type=int, dest="hidden_units", default=100)
    parser.add_argument("--hidden2", type=int, dest="hidden2_units", default=0)
    parser.add_argument("--kb", type=int, dest="window_b", default=1)
    parser.add_argument("--k1", type=int, dest="window_l1", default=3)
    parser.add_argument("--k2r", type=int, dest="window_l2r", default = 1)
    parser.add_argument("--k2l", type=int, dest="window_l2l", default = 1)  
    parser.add_argument("--lr", type=float, dest="learning_rate", default=0.1)
    parser.add_argument("--outdir", type=str, dest="output", default="results")
    parser.add_argument("--activation", type=str, dest="activation", default="tanh")
    parser.add_argument("--optimizer",type=str, dest="optimizer", default="adam")
    parser.add_argument("--lstmlayers", type=int, dest="lstm_layers", default=2)
    parser.add_argument("--lstmdims", type=int, dest="lstm_dims", default=200)
    parser.add_argument("--dynet-seed", type=int, dest="seed", default=7)
    parser.add_argument("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_argument("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_argument("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_argument("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_argument("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_argument("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_argument("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_argument("--dynet-mem", type=int, dest="cnn_mem", default=512)
    

    args = parser.parse_args()
        
    config = yaml.safe_load(open("configuration.yml"))
    print config       
    print config["udpipe"]
    
    #TODO load lookup table for languages?
    #PARSING WITH NEURAL COVINGTON
    
    if not args.predictFlag:
        print "Training..."
        #TRAINING PHASE
        if not (args.rlFlag or args.rlMostFlag or args.headFlag):
            print 'You must use either --userlmost or --userl or --usehead (you can use multiple)'
            sys.exit()

        print 'Preparing vocab'
        words, w2i, lemmas, l2i, cpos, pos, feats, rels = bcovington.utils.vocab(args.conll_train)

        print "lysparse-cpos", cpos
        print "lysparse-pos", pos

        with open(os.path.join(args.output, args.params), 'w') as paramsfp:
            pickle.dump((words, w2i, lemmas, l2i, cpos, pos, rels, args), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing blstm covington:'
        parser = covington.CovingtonBILSTM(words, lemmas, cpos, pos, rels, w2i, l2i, args)

        for epoch in xrange(args.epochs):
            print 'Starting epoch', epoch
            parser.Train(args.conll_train)
            devpath = os.path.join(args.output, 'dev_epoch_' + str(epoch+1) + '.conll')
            bcovington.utils.write_conll(devpath, parser.Predict(args.conll_dev))
            os.system('perl /home/david.vilares/Software/MaltOptimizer-1.0.3/eval.pl -g ' + args.conll_dev + ' -s ' + devpath  + ' > ' + devpath + '.txt &')
            print 'Finished predicting dev'
            parser.Save(os.path.join(args.output, args.model + str(epoch+1)))
    else:
        print "Predicting... "
        
        if args.input == None:
            raise ValueError("--input must contain a valid path when used --predict")
        
        #Loaded a pipeline object
        if args.pipe == PIPELINE_UDPIPE:
            pipe = utils.UDPipe(UDPIPE_MODEL, config[YAML_UDPIPE])        
        
        #Reading
        if INPUT_RAW == args.input_type:
            raw_content = utils.read_raw_file(args.input)
            conllu = pipe.run(raw_content)
        elif INPUT_CONLLU == args.input_type:
            raise NotImplementedError("TODO: Implement read_conllu_file()")
        else:
            raise NotImplementedError("--input_type "+args.input_type+" not supported")
        
        print conllu
        
        f_temp = tempfile.NamedTemporaryFile("w")
        f_temp.write(conllu)
        
        #TEST PHASE
        with open(args.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = args.external_embedding

        parser = covington.CovingtonBILSTM(words, lemmas, cpos, pos, rels, w2i, l2i, stored_opt)
        parser.Load(args.model)
        
        
        testpath = f_temp.name #os.path.join(args.output, 'test_pred.conll')
        print "testpath", testpath
        ts = time.time()
        pred = list(parser.Predict(testpath))
        te = time.time()
        utils.write_conll(testpath, pred)
        os.system('perl /home/david.vilares/Software/MaltOptimizer-1.0.3/eval.pl -g ' + args.conll_test + ' -s ' + testpath  + ' > ' + testpath + '.txt &')
        print 'Finished predicting test',te-ts
    
    
    
    
    
    
    
    
    
    