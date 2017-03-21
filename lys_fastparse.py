from argparse import ArgumentParser
from lysfastparse import utils


"""
Example of execution

python lys_fastparse.py \
--input  /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Spanish/es-ud-train.txt \
--input_type raw
"""

#INPUT TYPES
INPUT_RAW = "raw"
INPUT_CONLLU = "conllu"

#AVAILABLE PIPELINES
PIPELINE_UDPIPE = "UDpipe"

#TODO: Polish this
UDPIPE_MODEL = "/data/UDpipe/models/gl_udv2"

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to the input file")
    parser.add_argument("--input_type", dest="input_type",help="Style of the input file [raw|conllu]")
    parser.add_argument("--pipe", dest="pipe",default="UDpipe",help="Framework used to do the pipeline. Only \"UDpipe\" supported")
    args = parser.parse_args()
    
    #Loaded a pipeline object
    if args.pipe == PIPELINE_UDPIPE:
        pipe = utils.UDPipe(UDPIPE_MODEL)
        
    
    
    #TODO load lookup table for languages?
    print pipe
    
    #Reading
    if INPUT_RAW == args.input_type:
        raw_content = utils.read_raw_file(args.input)
        conllu = pipe.run(raw_content)
    elif INPUT_CONLLU == args.input_type:
        raise NotImplementedError("TODO: Implement read_conllu_file()")
    else:
        raise NotImplementedError("--input_type "+args.input_type+" not supported")