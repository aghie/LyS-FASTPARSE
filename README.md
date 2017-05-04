# LyS-FASTPARSE at CoNLL2017 shared task on end-to-end dependency parsing

## A BIST-parser for non-projective transition-based dependency parsing

A BI-LSTM implementation of a transition-based Covington's algorithm with dynamic oracle for unrestricted dependency parsing.

For segmentation and part-of-speech tagging the current system relies on the output provided by UDpipe.


## Required software

Python 2.7 interpreter

[Dynet python module](http://dynet.readthedocs.io/en/latest/python.html)

[Gensim](https://radimrehurek.com/gensim/)

[tarjan](https://pypi.python.org/pypi/tarjan/)


If you want to run our released pretrained models, make sure your BOOST version is 1.54. There is currently a dependency between the saved models and the version of BOOST that Dynet uses, so you might not be able to load the trained models if such version is different.

## Usage

### How to train a parser

The command used to train the Covington parsers is pretty similar to the one of the original BIST-parsers, but we optionally included fine PoStags and feats as external embeddings too:

	python lys_fastparse.py \
	--dynet-seed 123456789 \
	--dynet-mem $DYNET_MEM \
	--epochs 30 \
	--model $NAME_OUTPUT_MODEL \
	--conf $PATH_TO_configuration.yml \
	--outdir $OUTDIR \
	--train $PATH_TO_TRAIN_CONLLU_FILE \
	--dev $PATH_TO_DEV_CONLLU_FILE \
	--k1 $K1 \
	--kb $KB \
	--lstmdims $LSTMDIMS \
	--wembedding $WEMBEDDING \
	--extrn $PATH_EMBEDDINGS \
	--extrn_pos $PATH_POS_EMBEDDINGS \
	--extrn_feats $PATH_FEATS_EMBEDDINGS \
	--userl \
	--usehead \

For a description of all available options type:

	python lys_fastparse.py --help 

To run a trained model:

TBA

To execute the trained models using the official CoNLL/TIRA data directory use:

TBA



## Citation

TBA

## License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## References

Kiperwasser, E., & Goldberg, Y. (2016). Simple and accurate dependency parsing using bidirectional LSTM feature representations. arXiv preprint arXiv:1603.04351.

Straka, M., Hajic, J., & Straková, J. (2016). UD-Pipe: Trainable pipeline for processing CoNLL-U files performing tokenization, morphological analysis, POS tagging and parsing. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016).

Gómez-Rodríguez, C., & Fernández-González, D. (2015). An efficient dynamic oracle for unrestricted non-projective parsing. Volume 2: Short Papers, 256.

Nivre, J. (2008). Algorithms for deterministic incremental dependency parsing. Computational Linguistics, 34(4), 513-553.

Covington, M. A. (2001). A fundamental algorithm for dependency parsing. In Proceedings of the 39th annual ACM southeast conference (pp. 95-102).



