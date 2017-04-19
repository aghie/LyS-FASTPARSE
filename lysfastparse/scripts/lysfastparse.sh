#!/bin/bash
K1=1
KB=3
LSTMDIMS=128
TREEBANK=UD_Basque
BASE_SOURCE=/home/david.vilares/git/LyS-FASTPARSE
BASE_DIR_DATA="/data/david.vilares/ud-treebanks-conll2017-proof/"
OUTDIR=/home/david.vilares/Escritorio/Papers/bist-covington
#PATH_EMBEDDINGS=/data/david.vilares/FB_embeddings/wiki.hr.vec
EMBS="fb"
EMBS_CPOS="ud"
EMBS_POS="ud"
EMBS_FEATS="ud"


if [ EMBS="fb" ]; then
	PATH_EMBEDDINGS=/data/david.vilares/UD_embeddings/$TREEBANK
else
	PATH_EMBEDDINGS=None
fi

if [ EMBS_CPOS="ud" ]; then
	PATH_CPOS_EMBEDDINGS=/data/david.vilares/UD_CPOS_embeddings/$TREEBANK
else
	PATH_CPOS_EMBEDDINGS=None
fi

if [ EMBS_POS="ud" ]; then
	PATH_POS_EMBEDDINGS=/data/david.vilares/UD_POS_embeddings/$TREEBANK
else
	PATH_POS_EMBEDDINGS=None
fi

if [ EMBS_FEATS="ud" ]; then
	PATH_FEATS_EMBEDDINGS=/data/david.vilares/UD_FEATS_embeddings/$TREEBANK
else
	PATH_FEATS_EMBEDDINGS=None
fi

python $BASE_SOURCE/lys_fastparse.py \
--dynet-seed 123456789 \
--dynet-mem 6000 \
--input_type raw \
--epochs 15 \
--conf $BASE_SOURCE/configuration.yml \
--outdir $OUTDIR/$TREEBANK-k1_$K1-kb_$KB-lstmdims_$LSTMDIMS-emb-$EMBS \
--train $BASE_DIR_DATA/$TREEBANK/eu-ud-train.conllu \
--dev $BASE_DIR_DATA/$TREEBANK/eu-ud-dev.conllu \
--test $BASE_DIR_DATA/$TREEBANK/eu-ud-test.conllu \
--k1 $K1 \
--kb $KB \
--lstmdims $LSTMDIMS \
--extrn $PATH_EMBEDDINGS \
--extrn_pos $PATH_POS_EMBEDDINGS \
--extrn_cpos $PATH_CPOS_EMBEDDINGS \
--extrn_feats $PATH_FEATS_EMBEDDINGS \
--userl \
--usehead \
#> /home/david.vilares/Papers/bist-covington/log/$TREEBANK-k1_$K1-kb_$KB-lstmdims_$LSTMDIMS-emb-$EMBS_embcpos-$EMB_CPOS.log 2>&1 & 

