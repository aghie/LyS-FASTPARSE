from dynet import *
from utils_bcovington import read_conll, write_conll, CovingtonConfiguration
from operator import itemgetter
from itertools import chain
from tarjan import tarjan
import time, random
import numpy as np
import os
import warnings


"""
This is a module extended from original transition-based BIST-Parser:

https://github.com/elikip/bist-parser/blob/master/barchybrid/
Kiperwasser, E., & Goldberg, Y. (2016). Simple and accurate dependency parsing using bidirectional LSTM feature representations. arXiv preprint arXiv:1603.04351.


that has been adapted to include to support non-projective transition-based dependency parsing
using an implementation (O(n^2)) of the traditional Covington's (2001) algorithm, according
to the list-based transition-based described in Nivre (2008).

Covington, M. A. (2001). A fundamental algorithm for dependency parsing. In Proceedings of the 39th annual ACM southeast conference (pp. 95-102).
Nivre, J. (2008). Algorithms for deterministic incremental dependency parsing. Computational Linguistics, 34(4), 513-553.

We also include the O(n) dynamic oracle described in Gomez-Rodriguez and Fernandez-Gonzalez (2015).
TODO: Current implementation is O(n^2)

Gomez-Rodriguez, C., & Fernandez-Gonzalez, D. (2015). An efficient dynamic oracle for unrestricted non-projective parsing. Volume 2: Short Papers, 256.

"""



class CovingtonBILSTM:
    
    #ACTIVATION FUNCTIONS
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    RELU = 'relu'
    TANH3 = 'tanh3'
    
    #OPTIMIZERS
    SGD="sgd"
    MOMENTUM="momentum"
    ADAGRAD="adagrad"
    ADADELTA="adadelta"
    ADAM = "adam"
      
    #SPECIAL INDEXES
    INDEX_WORD_PAD = 1
    INDEX_WORD_INITIAL = 2
    INDEX_POS_PAD = 1
    INDEX_POS_INITIAL = 2
    INIT_WORD_INDEX = 3
    INIT_POS_INDEX = INIT_WORD_INDEX
    
    INDEX_FEATS_PAD = 1
    INDEX_FEATS_INITIAL= 2
    INIT_FEATS_INDEX = INIT_WORD_INDEX
    
    #TRANSITIONS
    LEFT_ARC = 0
    RIGHT_ARC = 1
    SHIFT = 2
    NO_ARC = 3
    TRANSITIONS = [LEFT_ARC, RIGHT_ARC, SHIFT, NO_ARC]    

    #OTHER HYPERPARAMETERS
    SIZE_TRANSITIONS = len(TRANSITIONS)
    
    def __init__(self, words, lemmas, cpos, pos, feats, rels, w2i, l2i, options, path_oov_external_embedding=None,
                 pretrained=False):
        
        self.model = Model()
        if options.optimizer == self.ADAM:
            self.trainer = AdamTrainer(self.model)
        elif options.optimizer == self.SGD:
            self.trainer = SimpleSGDTrainer(self.model)
        elif options.optimizer == self.MOMENTUM:
            self.trainer = MomentumSGDTrainer(self.model)
        elif options.optimizer == self.ADAGRAD:
            self.trainer = AdagradTrainer(self.model)
        elif options.optimizer == self.ADADELTA:
            self.trainer = AdadeltaTrainer(self.model)
        else:
            raise NotImplementedError("Selected optimizer is not available")
                     
        random.seed(1)

        self.activations = {self.TANH: tanh, 
                            self.SIGMOID: logistic, 
                            self.RELU: rectify, 
                            self.TANH3: (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        
        self.activation = self.activations[options.activation]

        self.oracle = options.oracle
        
        
        self.ldims = options.lstm_dims * 2 #*2 because it is a bi-lstm
        self.wdims = options.wembedding_dims 
        self.pdims = options.pembedding_dims 
        self.rdims = options.rembedding_dims 
        self.layers = options.lstm_layers
        self.wordsCount = words
        
        self.vocab = {word: ind+self.INIT_WORD_INDEX for word, ind in w2i.iteritems()} 
        self.lemmas = {lemma: ind+self.INIT_WORD_INDEX for lemma,ind in l2i.iteritems()}
        self.cpos = {cpos: ind+self.INIT_POS_INDEX for ind, cpos in enumerate(cpos)}
        self.pos = {pos: ind+self.INIT_POS_INDEX for ind, pos in enumerate(pos)}
        self.feats = {f: ind+self.INIT_FEATS_INDEX for ind, f in enumerate(feats)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        
        #List of dependency types
        self.irels = rels 

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.kb = options.window_b
        self.kl1 = options.window_l1
        self.kl2_r = options.window_l2r
        self.kl2_l = options.window_l2l

        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)
        
        #Reading external embedding files, if they exists

        #INFORMATION FOR EXTERNAL WORD EMBEDDINGS
        self.external_embedding = None
        self.edim = None
        self.noextrn = None
        self.extrnd = None
        self.elookup = None
        if options.external_embedding is not None and os.path.exists(options.external_embedding):
            self.external_embedding, self.edim,self.noextrn,self.extrnd, self.elookup = self._assign_external_embeddings(options.external_embedding,
                                                                                                                    self.INDEX_WORD_PAD, self.INDEX_WORD_INITIAL)
        else:
            warnings.warn("Not using any external file for FORM embeddings")
                
        #INFORMATION FOR THE EXTERNAL CPOSTAG EMBEDDINGS
        self.cpos_external_embedding = None
        self.cpos_edim = None
        self.cpos_noextrn = None
        self.cpos_extrnd = None
        self.cpos_elookup = None
        if options.cpos_external_embedding is not None and os.path.exists(options.cpos_external_embedding):
            self.cpos_external_embedding, self.cpos_edim,self.cpos_noextrn,self.cpos_extrnd, self.cpos_elookup = self._assign_external_embeddings(options.cpos_external_embedding,
                                                                                                                                             self.INDEX_POS_PAD, self.INDEX_POS_INITIAL)
        else:
            warnings.warn("Not using any external file for CPOSTAG embeddings")
            
        #INFORMATION FOR THE EXTERNAL POSTAG EMBEDDINGS
        self.pos_external_embedding = None
        self.pos_edim = None
        self.pos_noextrn = None
        self.pos_extrnd = None
        self.pos_elookup= None
        if options.pos_external_embedding is not None and os.path.exists(options.pos_external_embedding):
            self.pos_external_embedding, self.pos_edim,self.pos_noextrn,self.pos_extrnd, self.pos_elookup = self._assign_external_embeddings(options.pos_external_embedding,
                                                                                                                                             self.INDEX_POS_PAD, self.INDEX_POS_INITIAL)
        else:
            warnings.warn("Not using any external file for POSTAG embeddings")
            
        #INFORMATION FOR THE EXTERNAL FEATS EMBEDDINGS
        self.feats_external_embedding = None
        self.feats_edim = None
        self.feats_noextrn = None
        self.feats_extrnd = None
        self.feats_elookup= None
              
        if options.feats_external_embedding is not None and os.path.exists(options.feats_external_embedding):
            self.feats_external_embedding, self.feats_edim,self.feats_noextrn,self.feats_extrnd, self.feats_elookup = self._assign_external_embeddings(options.feats_external_embedding,                                                                                                                        self.INDEX_FEATS_PAD, self.INDEX_FEATS_INITIAL)
        else:
            warnings.warn("Not using any external file for FEATS embeddings")        
        
        
        #INFORMATION FOR THE EXTERNAL FEATS EMBEDDINGS
#         self.lemmas_external_embedding = None
#         self.lemmas_edim = None
#         self.lemmas_noextrn = None
#         self.lemmas_extrnd = None
#         self.lemmas_elookup= None
              
#         if options.lemmas_external_embedding is not None and os.path.exists(options.lemmas_external_embedding):
#             self.lemmas_external_embedding, self.lemmas_edim,self.lemmas_noextrn,self.lemmas_extrnd, self.lemmas_elookup = self._assign_external_embeddings(options.lemmas_external_embedding,                                                                                                                        self.INDEX_FEATS_PAD, self.INDEX_FEATS_INITIAL)
#         else:
#             warnings.warn("Not using any external file for LEMMAS embeddings")        
        
        
        
        
        self.oov_external_embedding = None
        self.oov_edim = None
        self.oov_noextrn = None
        self.oov_extrnd = None
        self.oov_elookup = None
        
        
        if path_oov_external_embedding is not None and os.path.exists(options.feats_external_embedding):
                        self.oov_external_embedding, self.oov_edim,self.oov_noextrn,self.oov_extrnd, self.oov_elookup = self._assign_external_embeddings(path_oov_external_embedding,
                                                                                                                    self.INDEX_WORD_PAD, self.INDEX_WORD_INITIAL) 

        if self.oov_external_embedding is not None and self.oov_edim != self.edim:
            raise ValueError("The dimensions of the embeddings for OOV words is not equal to the dimension of the rest of external word embeddings (self.oov_edim != self.edim)")
            
        #Obtaining the dimension of the input
        dims = (self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0) + 
                                          (self.cpos_edim if self.cpos_external_embedding is not None else 0) +
                                          (self.pos_edim if self.pos_external_embedding is not None else 0)+
                                          (self.feats_edim if self.feats_external_embedding is not None else 0)
#                                           +
#                                           (self.lemmas_edim if self.lemmas_external_embedding is not None else 0)
                                          )
        
        
        #Initialization of the architecture
        
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.surfaceBuilders = [VanillaLSTMBuilder(1, dims, self.ldims * 0.5, self.model),
                                    VanillaLSTMBuilder(1, dims, self.ldims * 0.5, self.model)]
            self.bsurfaceBuilders = [VanillaLSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model),
                                     VanillaLSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [VanillaLSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model), LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model)]
            else:
                self.surfaceBuilders = [SimpleRNNBuilder(1, dims, self.ldims * 0.5, self.model), LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]


        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = self.INDEX_WORD_PAD
        self.cpos['*PAD*'] = self.INDEX_POS_PAD
        self.feats['*PAD*'] = self.INDEX_FEATS_PAD

        self.vocab['*INITIAL*'] = self.INDEX_WORD_INITIAL
        self.cpos['*INITIAL*'] = self.INDEX_POS_INITIAL
        self.feats['*INITIAL*'] = self.INDEX_FEATS_INITIAL

        self.wlookup = self.model.add_lookup_parameters((len(words) + self.INIT_WORD_INDEX, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(cpos) + self.INIT_POS_INDEX, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        
        self.word2lstm = self.model.add_parameters((self.ldims, dims))
        
        self.word2lstmbias = self.model.add_parameters((self.ldims))
        self.lstm2lstm = self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias = self.model.add_parameters((self.ldims))

        self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * (self.kl1 + self.kl2_l + self.kl2_r  + self.kb)))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((self.SIZE_TRANSITIONS, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias = self.model.add_parameters((self.SIZE_TRANSITIONS))

        self.rhidLayer = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * (self.kl1 + self.kl2_l + self.kl2_r  + self.kb)))
        self.rhidBias = self.model.add_parameters((self.hidden_units))

        self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

        self.routLayer = self.model.add_parameters((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias = self.model.add_parameters((2 * (len(self.irels) + 0) + 1))

        self.pretrained = pretrained


    def _assign_external_embeddings(self,option_external_embedding,
                                    index_pad,index_initial):
        """
        Reads an external embedding file
        Returns:
        external_embedding: A dictionary of key:embedding
        edim: Dimension of the embedding
        noextrn: ??
        extrnd: Index for each key
        elookup: Parameter lookup 
        """
            

        if option_external_embedding is not None:
 
            external_embedding_fp = open(option_external_embedding,'r')
            external_embedding_fp.readline()
                
            external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] 
                                           for line in external_embedding_fp}
            
            
            external_embedding_fp.close()
    
            edim = len(external_embedding.values()[0])
            noextrn = [0.0 for _ in xrange(edim)]
            extrnd = {element: i + self.INIT_POS_INDEX 
                                    for i, element in enumerate(external_embedding)}
            elookup = self.model.add_lookup_parameters((len(external_embedding) + self.INIT_WORD_INDEX, edim))
                
            for element, i in extrnd.iteritems():
                    elookup.init_row(i, external_embedding[element])
            extrnd['*PAD*'] = index_pad
            extrnd['*INITIAL*'] = index_initial

            return external_embedding, edim, noextrn, extrnd, elookup
    
        return None,None,None,None,None



    def __evaluate(self, c, train):
        """
        @param c: A CovingtonConfiguration instance
        @param train: True if used in the training phase, False otherwise
        Returns the scores for all possible transitions (training)
        or the top ones (testing) for a given configuration c
        """
              
        #Gets the embeddings for the terms to be used in the prediction
        top_l1  = [c.sentence[c.l1-i].lstms if c.l1 - i > 0 else [self.empty] for i in xrange(self.kl1)]
        top_l2l = [c.sentence[c.l1+1+i].lstms if c.l1+1+i < c.b  else [self.empty] for i in xrange(self.kl2_l)]
        top_l2r = [c.sentence[c.b-i].lstms if c.b-i > c.l1 else [self.empty] for i in xrange(self.kl2_r)]
        topBuffer = [c.sentence[c.b+i-1].lstms if c.b+i-1 <= c.sentence[-1].id else [self.empty] for i in xrange(self.kb)]

        input = concatenate(list(chain(*(top_l1 + top_l2l + top_l2r + topBuffer))))

        if self.hidden2_units > 0:
            routput = (self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(self.rhidLayer.expr() * input + self.rhidBias.expr())) + self.routBias.expr())
        else:
            routput = (self.routLayer.expr() * self.activation(self.rhidLayer.expr() * input + self.rhidBias.expr()) + self.routBias.expr())

        if self.hidden2_units > 0:
            output = (self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr())) + self.outBias.expr())
        else:
            output = (self.outLayer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr()) + self.outBias.expr())

        scrs, uscrs = routput.value(), output.value()

        if train:
            left_arc_info = [(rel,self.LEFT_ARC, scrs[1+j*2] + uscrs[self.LEFT_ARC], routput[1+j*2]+ output[self.LEFT_ARC]) 
                                for j, rel in enumerate(self.irels) if c.l1 > 0 and c.l1 < c.b and c.b <= c.sentence[-1].id]
    
            right_arc_info = [(rel,self.RIGHT_ARC, scrs[2+j*2] + uscrs[self.RIGHT_ARC], routput[2+j*2]+ output[self.RIGHT_ARC]) 
                                 for j, rel in enumerate(self.irels) if c.l1 >= 0 and c.l1 < c.b and c.b <= c.sentence[-1].id]
            
            shift_info = [ (None, self.SHIFT, scrs[0] + uscrs[self.SHIFT], routput[0] + output[self.SHIFT]) ] if c.b <= c.sentence[-1].id else []

            no_arc_info = [(None, self.NO_ARC,scrs[3] + uscrs[self.NO_ARC], routput[3] + output[self.NO_ARC] )] if c.l1> 0 and  c.b <= c.sentence[-1].id else []
            
            ret = [left_arc_info,right_arc_info, shift_info, no_arc_info]
                            
        else:
            #It is done different from the 'train' phase, due to the dynamic oracle.
            #In the test phase we already pick the most likely transition/dependency instead of returning them all
            #and then selecting one according to the prediction of the dynamic oracle
            sLEFT,rLEFT = max(zip(scrs[1::2],self.irels))
            sRIGHT,rRIGHT = max(zip(scrs[2::2],self.irels))
            sLEFT += uscrs[self.LEFT_ARC]
            sRIGHT += uscrs[self.RIGHT_ARC]
            ret = [ [(rLEFT, self.LEFT_ARC, sLEFT) ] if (c.l1 > 0 and c.l1 < c.b and c.b <= c.sentence[-1].id and self._is_valid_left_arc(c)) else [],  
                    [(rRIGHT, self.RIGHT_ARC, sRIGHT) ] if (c.l1 >= 0 and c.l1 < c.b and c.b <= c.sentence[-1].id and self._is_valid_right_arc(c)) else [],  
                    [(None, self.SHIFT, scrs[0] + uscrs[self.SHIFT]) ] if (c.b <= c.sentence[-1].id) else [],
                    [(None, self.NO_ARC,scrs[3] + uscrs[self.NO_ARC]) ] if (c.l1 > 0 and c.b <= c.sentence[-1].id) else []
                     ]
        return ret


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        evec = self.elookup[1] if self.external_embedding is not None  else None
        cpos_evec = self.cpos_elookup[1] if self.cpos_external_embedding is not None else None
        pos_evec = self.pos_elookup[1] if self.pos_external_embedding is not None else None
        feats_evec = self.feats_elookup[1] if self.feats_external_embedding is not None else None
      #  lemmas_evec = self.lemmas_elookup[1] if self.lemmas_external_embedding is not None else None
        paddingWordVec = self.wlookup[1]
        paddingPosVec = self.plookup[1] if self.pdims > 0 else None
      #  paddingVec = tanh(self.word2lstm.expr() * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec, cpos_evec, pos_evec, feats_evec, lemmas_evec])) + self.word2lstmbias.expr())
        paddingVec = tanh(self.word2lstm.expr() * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec, cpos_evec, pos_evec, feats_evec])) + self.word2lstmbias.expr())
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])


    def getWordEmbeddings(self, sentence, train):
        """
        Gets the embeddings (also external) for every term in a sentence
        Returns a vector of all embeddings concatenated
        """
        
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))
            sys.stdout.flush()
            root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0]
            root.cposvec = self.plookup[int(self.cpos.get(root.cpos,0))] if self.pdims > 0 else None

            #For word embeddings
            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.norm]]
                else:
                    if (self.oov_external_embedding is not None and root.form.replace(" ","_") in self.oov_external_embedding):
                        root.evec = self.oov_elookup[self.oov_extrnd[root.form.replace(" ","_")]]
                    else:
                        root.evec = self.elookup[0]
            else:
                root.evec = None

            #For cpostag embeddings
            if self.cpos_external_embedding is not None:
                if root.cpos in self.cpos_external_embedding:
                    root.cposevec = self.cpos_elookup[self.cpos_extrnd[root.cpos]]
                else:
                    root.cposevec = self.cpos_elookup[0]
            else:
                root.cposevec = None
            
            #For postag embeddings
            if self.pos_external_embedding is not None:
                if root.pos in self.pos_external_embedding:
                    root.posevec = self.pos_elookup[self.pos_extrnd[root.pos]]
                else:
                    root.posevec = self.pos_elookup[0]
            else:
                root.posevec = None
#             
            #For feats embeddings
            if self.feats_external_embedding is not None:
                if root.feats in self.feats_external_embedding:
                    root.featsevec = self.feats_elookup[self.feats_extrnd[root.feats]]
                else:
                    root.featsevec = self.feats_elookup[0]
            else:
                root.featsevec = None
            
            
            #For lemmas embeddings
#             if self.lemmas_external_embedding is not None:
#                 if root.lemma in self.lemmas_external_embedding:
#                     root.lemmasevec = self.lemmas_elookup[self.lemmas_extrnd[root.lemma]]
#                 else:
#                     root.lemmasevec = self.lemmas_elookup[0]
#             else:
#                 root.lemmasevec = None            
            
            
         #   root.ivec = concatenate(filter(None, [root.wordvec, root.cposvec, root.evec, root.cposevec, root.posevec, root.featsevec, root.lemmasevec]))
            root.ivec = concatenate(filter(None, [root.wordvec, root.cposvec, root.evec, root.cposevec, root.posevec, root.featsevec]))
            
        if self.blstmFlag:
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence, reversed(sentence)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence:
                root.vec = concatenate( [root.fvec, root.bvec] )

            if self.bibiFlag:
                bforward  = self.bsurfaceBuilders[0].initial_state()
                bbackward = self.bsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence, reversed(sentence)):
                    bforward = bforward.add_input( froot.vec )
                    bbackward = bbackward.add_input( rroot.vec )
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence:
                    root.vec = concatenate( [root.bfvec, root.bbvec] )

        else:
            for root in sentence:
                root.ivec = (self.word2lstm.expr() * root.ivec) + self.word2lstmbias.expr()
                root.vec = tanh( root.ivec )


    def Predict(self, conll_path):
        """
        Makes non-projective depending parsing prediction given a ConLL-X file
        """
    
        
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                self.Init()

                l1 = sentence[0].id
                b = sentence[1].id
                arcs = set([])   
                
                self.getWordEmbeddings(sentence, False)

                for root in sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                c = CovingtonConfiguration(l1,b,sentence,arcs)
                while not self._is_final_state(b,sentence):

                    transition_scores = self.__evaluate(c, False)

                    
                    best = max(chain(*transition_scores), key = itemgetter(2) )

                    if best[1] == self.LEFT_ARC:
                        
                        sentence[l1].pred_parent_id = sentence[b].id
                        sentence[l1].pred_relation = best[0]
                        best_op = self.LEFT_ARC
                        if self.rlMostFlag:
                            sentence[b].lstms[best_op+hoffset] = sentence[l1].lstms[best_op+hoffset]
                        if self.rlFlag:
                            sentence[b].lstms[best_op+hoffset] = sentence[l1].vec

                        arcs.add((b,l1))
                        l1 = l1 -1
                        
                    elif best[1] == self.RIGHT_ARC:
                        
                        sentence[b].pred_parent_id = sentence[l1].id
                        sentence[b].pred_relation = best[0]

                        best_op = self.RIGHT_ARC
                        if self.rlMostFlag:
                            sentence[l1].lstms[best_op+hoffset] = sentence[b].lstms[best_op+hoffset]
                        if self.rlFlag:
                            sentence[l1].lstms[best_op+hoffset] = sentence[b].vec
                        
                        arcs.add((l1,b))
                        l1 = l1-1

                    elif best[1] == self.SHIFT:
                        l1 = b
                        b = b + 1


                    elif best[1] == self.NO_ARC:
                        l1 = l1 - 1

                    c = CovingtonConfiguration(l1,b,sentence,arcs)
                renew_cg()
                yield sentence


    def Train(self, conll_path):
        """
        Trains a O(n^2) Covington's parser with a O(n^2) dynamic oracle
        """
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ltotal = 0
        ninf = -float('inf')

        hoffset = 1 if self.headFlag else 0

        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            
            random.shuffle(shuffledData)


            errs = []
            eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):    
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start                  
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                self.getWordEmbeddings(sentence, True)
                #We obtain the gold arcs to then compute the dynamic oracle for covington
                gold_arcs = set([])
                for word in sentence:
                    
                    #TODO: Weird error if not, adds and arc (0,0)
                    if word.id != word.parent_id:
                        gold_arcs.add((word.parent_id,word.id))
                
                
                l1 = sentence[0].id
                b = sentence[1].id
                arcs = set([])               
                c = CovingtonConfiguration(l1,b,sentence,arcs)
                loss_c = self._loss(c,gold_arcs, iSentence)
            
                for word in sentence:
                    word.lstms = [word.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                while not self._is_final_state(b,sentence):

                    costs = [None,None,None,None]
                    transition_scores = self.__evaluate(c, True)

                    #We determine if the transitions are valid for a given configuration c
                    for t in self.TRANSITIONS:
                        
                        l1_aux = l1
                        b_aux = b
                        arcs_aux =  set(arcs)
                        valid_transition = False
                        
                        if t == self.LEFT_ARC and self._is_valid_left_arc(c):
                            arcs_aux.add((b_aux,l1_aux))
                            l1_aux = l1_aux -1
                            valid_transition = True

                        if t == self.RIGHT_ARC and l1 >=0 and self._is_valid_right_arc(c):
                            arcs_aux.add((l1_aux,b_aux))
                            l1_aux = l1_aux-1
                            valid_transition = True
                                 
                        if t == self.NO_ARC and l1 >0:
                            l1_aux = l1_aux-1
                            valid_transition = True   
                               
                        if t == self.SHIFT:
                            l1_aux = b_aux
                            b_aux = b_aux + 1 
                            valid_transition = True      
                        
                        if valid_transition:  
             
                            new_c = CovingtonConfiguration(l1_aux,b_aux,sentence,arcs_aux)
                            loss_new_c = self._loss(new_c,gold_arcs,iSentence)
                                               
                            cost = loss_new_c - loss_c
                            costs[t] = float(cost)

                    #Valid transitions are those with cost 0
                    #If it is a LEFT/RIGHT arc, also the relation must match with the one in gold standard
                    valid_transitions = [s for s in chain(*transition_scores) if costs[s[1]] == 0 and (s[1] in [self.SHIFT,self.NO_ARC] 
                                                                                                          or ((s[1] == self.LEFT_ARC and s[0] == sentence[l1].relation) 
                                                                                                          or (s[1] == self.RIGHT_ARC and s[0] == sentence[b].relation)))]

                    best_valid = max(valid_transitions, key=itemgetter(2))

                    wrong_transitions = [s for s in chain(*transition_scores) if costs[s[1]] is not None and ( (costs[s[1]] != 0) or (s[1] in [self.LEFT_ARC,self.RIGHT_ARC] 
                                                                                                          and ((s[1] == self.LEFT_ARC and s[0] != sentence[l1].relation) 
                                                                                                              or (s[1] == self.RIGHT_ARC and s[0] != sentence[b].relation))) ) ]
                    
                    #Aggressive exploration as done by Kiperwasser and Golberg (2016)
                    if wrong_transitions != []:
                        best_wrong = max(wrong_transitions, key=itemgetter(2))    

                        best = best_valid if ( (not self.oracle) or (best_valid[2] - best_wrong[2] > 1.0) 
                                              or (best_valid[2] > best_wrong[2] and random.random() > 0.1) ) else best_wrong
                    else:
                        best = best_valid 


                    #Moving a new configuration based on the "best" choice
                    if best[1] == self.LEFT_ARC:
                                      
                        sentence[l1].pred_parent_id = sentence[b].id
                        sentence[l1].pred_relation = best[0]

                        best_op = self.LEFT_ARC
                        if self.rlMostFlag:
                            sentence[b].lstms[best_op+hoffset] = sentence[l1].lstms[best_op+hoffset]
                        if self.rlFlag:
                            sentence[b].lstms[best_op+hoffset] = sentence[l1].vec
                        
                        child = sentence[l1]
                        arcs.add((b,l1))
                        l1 = l1 -1
                        
                    elif best[1] == self.RIGHT_ARC:
                        
                        
                        sentence[b].pred_parent_id = sentence[l1].id
                        sentence[b].pred_relation = best[0]

                        best_op = self.RIGHT_ARC
                        if self.rlMostFlag:
                            sentence[l1].lstms[best_op+hoffset] = sentence[b].lstms[best_op+hoffset]
                        if self.rlFlag:
                            sentence[l1].lstms[best_op+hoffset] = sentence[b].vec
                        
                        arcs.add((l1,b))
                        child = sentence[b]
                        l1 = l1-1


                    elif best[1] == self.SHIFT:
                        l1 = b
                        child = sentence[b]
                        b = b + 1


                    elif best[1] == self.NO_ARC:
                        l1 = l1 - 1
                        child = sentence[l1]


                    if best_valid[2] < best_wrong[2] + 1.0:
                        loss = best_wrong[3] - best_valid[3]
                        mloss += 1.0 + best_wrong[2] - best_valid[2]
                        eloss += 1.0 + best_wrong[2] - best_valid[2]
                        errs.append(loss)

                    
                    if best[1] not in [self.SHIFT, self.NO_ARC] and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                        lerrors += 1
                        if child.pred_parent_id != child.parent_id:
                            errors += 1 
                            eerrors += 1 

                    etotal += 1
                    c = CovingtonConfiguration(l1,b,sentence,arcs)
                    loss_c = self._loss(c,gold_arcs, iSentence)
                 

                if len(errs) > 50: 
                    eerrs = esum(errs)
                    scalar_loss = eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                    renew_cg()
                    self.Init()

        if len(errs) > 0:
            eerrs = (esum(errs)) # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence


    def _is_final_state(self,b,sentence):
        return b >= len(sentence)


    def _is_valid_left_arc(self,c):
        
        aux = set(c.A)
        aux.add((c.b,c.l1))
        l1_has_head = self._y_has_head(c.A, c.b, c.l1) 
        return (c.l1 > 0 and not l1_has_head
                and self._count_cycles(aux) == 0)


    def _is_valid_right_arc(self,c):
        
        b_has_head = self._y_has_head(c.A, c.l1, c.b)
        aux = set(c.A)
        aux.add((c.l1,c.b))
        return ((not b_has_head) and self._count_cycles(aux) == 0)
        
        
    """
    Gomez-Rodriguez & Fernandez-Gonzalez: 
    An Efficiente Dynamic Oracle for Unrestricted Non-Projective Parsing  (ACL,2015)
    Algorithm 1
    """
    def _loss(self, c, gold_arcs, iSentence):
        
        U = set([]) #set of unreachable nodes
        non_built_arcs = gold_arcs.difference(c.A)
        
        
        i = c.l1
        j = c.b
              
        for x,y in non_built_arcs: 
            left = min(x,y)  #O(n)
            right = max(x,y) #O(n)
            if (j > right or (j==right and i < left) or self._y_has_head(c.A,x,y)
                or self._weakly_connected(c.A, x, y,c, gold_arcs)):
                U.add((x,y))
        
        I = gold_arcs.difference(U)

        return len(U) + self._count_cycles( c.A.union(I))
    
    
    #TODO: This can be done more efficient
    #O(n^2)
    def _weakly_connected(self,A,x,y,c, gold_arcs):
        
        weakly_connected = False 
        end_path = False
        parent = x
        
        while parent != 0 and not weakly_connected and not end_path and  A  != set([]):
            if (parent,y) in A:
                weakly_connected = True
                break
            else:     

                for (a,b) in A:
                    if b == parent: 
                        parent = a
                        break
                    else:
                        end_path = True
                        
                    
        return weakly_connected
    
    
    """
    Tarjan (1972) implementation at https://github.com/bwesterb/py-tarjan/
    O(n)
    """
    def _count_cycles(self, A):
        
        d = {}
        for a,b in A:
            if a not in d:
                d[a] = [b]
            else:
                d[a].append(b)
                   
        return sum([1 for e in tarjan(d) if len(e) > 1])
    
    
    """
    Determines if node y has already a head
    """
    #O(n)
    def _y_has_head(self,A,x,y):
        
        for z,y_prime in A:
            if y_prime == y and z != x:
                return True
        return False
     
    #O(n)
#     def violates_single_root(self, A):
#         print A,[1 for (h,d) in A if h==0], len([1 for (h,d) in A if h==0]) != 0 
#         return len([1 for (h,d) in A if h==0]) != 0 
    