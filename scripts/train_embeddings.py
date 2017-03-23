
import gensim
import os
import codecs
from argparse import ArgumentParser


"""

Example:

python train_embeddings.py \
--input_conll /data/Universal\ Dependencies\ 2.0/ud-treebanks-conll2017/UD_Spanish/es-ud-train.conllu \
--column_to_embed 3 

"""



def read_conll_to_raw(path_conll,column):
    
    sentences =[]
    tokens = []
    with codecs.open(path_conll) as f_conll:
        lines = f_conll.readlines()
        for line in lines:
            tok = line.strip('\n').split()
            
            if len(tok)>1 and not line.startswith('#'):
                
                try:
                    int(tok[0]) #not to include multiword expressions
                  #  if tok[column] == "_": print line; exit()
                    tokens.append(tok[column])
                except ValueError:
                    continue
            else:
                sentences.append(tokens)
                tokens = []

    return sentences

def train_embeddings(sentences,path_output,size=50):
    model = gensim.models.Word2Vec(sentences,sg=1, size=size, window=10, negative=5, hs=0,
                                    sample= 0.1, iter=15, min_count=2)
    model.train(sentences)
    model.wv.save_word2vec_format(path_output)
    
    print model.wv.most_similar(positive=["SCONJ"])


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input_conll", dest="input_conll", help="Path to the input file",default=None)
    parser.add_argument("--input_dir", dest="input_dir", help="Path to a dir containing UD directories",default=None)
    parser.add_argument("--column_to_embed", dest="column_to_embed",type=int,help="Column from the CoNLLU file to be learned as embeddings", default="1",required=True)
    parser.add_argument("--output_dir", dest="output_dir",help="Path to the output dir", default="/tmp/")
    parser.add_argument("--size_embedding",dest="size_embedding", type=int, help="Desired size of embeddings", default=50)
    args = parser.parse_args()
    column = args.column_to_embed
    path_output = args.output_dir
    size_e = args.size_embedding
    
    
    if args.input_dir is not None:
        path_treebanks = args.input_dir
        treebank_paths = [path_treebanks+os.sep+f for f in os.listdir(path_treebanks)]
        
        for treebank_path in treebank_paths:
    
            name_treebank = treebank_path.split("/")[-1]
            treebank_files = [treebank_path+os.sep+f for f in os.listdir(treebank_path) 
                                  if f.endswith("ud-train.conllu") or f.endswith("ud-dev.conllu") or f.endswith("ud-dev.conllu")]
            sentences = []
            for path in treebank_files:
                sentences.extend(read_conll_to_raw(path,column)) 
                
            train_embeddings(sentences, path_output+os.sep+name_treebank+"_c-"+str(column),size_e)
    
    elif args.input_conll is not None:
        name_treebank = args.input_conll.split("/")[-2]
        print name_treebank
        path_conll = args.input_conll
        sentences = read_conll_to_raw(path_conll,column)        
        train_embeddings(sentences, path_output+name_treebank,size_e)

    else: 
        raise ValueError("--input_conll or --input_dir must be activated")    
    
    