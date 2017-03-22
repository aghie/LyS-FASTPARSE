# LyS-FASTPARSE

## Usage

Example (it trains a neural covington)

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

## TODO

TODO: Add support to select the right UDpipe model (now it is hard-coded)
TODO: It seems to be a problem with the sentence tokenization 
