#from pycnn import *
from dynet import *
from utils import ParseForest, read_conll, write_conll, CovingtonConfiguration, get_gold_arcs
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np
from tarjan import tarjan
import copy
import os
from geopy.units import arcsec



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
    
    #TRANSITIONS
    LEFT_ARC = 0
    RIGHT_ARC = 1
    SHIFT = 2
    NO_ARC = 3
    TRANSITIONS = [LEFT_ARC, RIGHT_ARC, SHIFT, NO_ARC]    

    #OTHER HYPERPARAMETERS
    SIZE_TRANSITIONS = len(TRANSITIONS)
    
    def __init__(self, words, lemmas, cpos, pos, rels, w2i, l2i, options):
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
        
        print self.wdims
        print self.pdims
        print self.rdims

        self.vocab = {word: ind+self.INIT_WORD_INDEX for word, ind in w2i.iteritems()} 
        self.cpos = {cpos: ind+self.INIT_POS_INDEX for ind, cpos in enumerate(cpos)}
        self.pos = {pos: ind+self.INIT_POS_INDEX for ind, pos in enumerate(pos)}
        #TODO : Do we need special indexes for feats too
        #self.feats = {feats: }
        self.rels = {word: ind for ind, word in enumerate(rels)}
        
        print "cpos", cpos
        print "pos", pos
        print "self.cpos", self.cpos
        print "self.pos", self.pos
     #   print "self.feats", self.feats

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

        #INFORMATION FOR EXTERNAL WORD EMBEDDINGS
        self.external_embedding = None
        self.edim = None
        self.noextrn = None
        self.extrnd = None
        self.elookup = None
        self.external_embedding, self.edim,self.noextrn,self.extrnd, self.elookup = self._assign_external_embeddings(options.external_embedding,
                                                                                                                    self.INDEX_WORD_PAD, self.INDEX_WORD_INITIAL)
                
        #INFORMATION FOR EXTERNAL CPOSTAG EMBEDDING
        self.cpos_external_embedding = None
        self.cpos_edim = None
        self.cpos_noextrn = None
        self.cpos_extrnd = None
        self.cpos_elookup = None
        self.cpos_external_embedding, self.cpos_edim,self.cpos_noextrn,self.cpos_extrnd, self.cpos_elookup = self._assign_external_embeddings(options.cpos_external_embedding,
                                                                                                                                             self.INDEX_POS_PAD, self.INDEX_POS_INITIAL)
                



        #TODO: Try to factor this
        #For external word embeddings
#         self.external_embedding = None
#         if options.external_embedding is not None:
#             external_embedding_fp = open(options.external_embedding,'r')
#             external_embedding_fp.readline()
#             self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] 
#                                        for line in external_embedding_fp}
#             external_embedding_fp.close()
# 
#             self.edim = len(self.external_embedding.values()[0])
#             self.noextrn = [0.0 for _ in xrange(self.edim)]
#             self.extrnd = {word: i + self.INIT_WORD_INDEX for i, word in enumerate(self.external_embedding)}
#             self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + self.INIT_WORD_INDEX, self.edim))
#             
#             for word, i in self.extrnd.iteritems():
#                  self.elookup.init_row(i, self.external_embedding[word])
#             self.extrnd['*PAD*'] = self.INDEX_WORD_PAD
#             self.extrnd['*INITIAL*'] = self.INDEX_WORD_INITIAL
# 
#             print 'Load external embedding. Vector dimensions', self.edim
            
#         #For external Cpostag embeddings
#         self.cpos_external_embedding = None
#         if options.cpos_external_embedding is not None:
#             
#             cpos_external_embedding_fp = open(options.cpos_external_embedding,'r')
#             cpos_external_embedding_fp.readline()
#             self.cpos_external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] 
#                                        for line in cpos_external_embedding_fp}
#             cpos_external_embedding_fp.close()
# 
#             self.cpos_edim = len(self.cpos_external_embedding.values()[0])
#             self.cpos_noextrn = [0.0 for _ in xrange(self.cpos_edim)]
#             self.cpos_extrnd = {cpostag: i + self.INIT_POS_INDEX 
#                                 for i, cpostag in enumerate(self.cpos_external_embedding)}
#             self.cpos_elookup = self.model.add_lookup_parameters((len(self.cpos_external_embedding) + self.INIT_POS_INDEX, self.cpos_edim))
#             
#             for cpostag, i in self.cpos_extrnd.iteritems():
#                  self.cpos_elookup.init_row(i, self.cpos_external_embedding[cpostag])
#             self.cpos_extrnd['*PAD*'] = self.INDEX_POS_PAD
#             self.cpos_extrnd['*INITIAL*'] = self.INDEX_POS_INITIAL
# 
#             print 'Load cpostag external embedding. Vector dimensions', self.cpos_edim
            
            
        
                      

        #dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        dims = (self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0) + 
                                          (self.cpos_edim if self.cpos_external_embedding is not None else 0))
        
        
        print self.ldims
        print self.ldims*0.5
        print ("dims",dims)
        
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

        self.vocab['*INITIAL*'] = self.INDEX_WORD_INITIAL
        self.cpos['*INITIAL*'] = self.INDEX_POS_INITIAL

        self.wlookup = self.model.add_lookup_parameters((len(words) + self.INIT_WORD_INDEX, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(cpos) + self.INIT_POS_INDEX, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        
        #self.word2lstm = self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
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



    def _assign_external_embeddings(self,option_external_embedding,
                                    index_pad,index_initial):
            
        print option_external_embedding
        if option_external_embedding is not None:
            
            print "Entra _assign_external_embedding"
                
            external_embedding_fp = open(option_external_embedding,'r')
            external_embedding_fp.readline()
                
            external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] 
                                           for line in external_embedding_fp}
            external_embedding_fp.close()
    
    
    
            edim = len(external_embedding.values()[0])
            noextrn = [0.0 for _ in xrange(edim)]
            extrnd = {element: i + self.INIT_POS_INDEX 
                                    for i, element in enumerate(external_embedding)}
            elookup = self.model.add_lookup_parameters((len(external_embedding) + self.INIT_POS_INDEX, edim))
                
            for element, i in extrnd.iteritems():
                    elookup.init_row(i, external_embedding[element])
            extrnd['*PAD*'] = index_pad
            extrnd['*INITIAL*'] = index_initial

            print 'Load cpostag external embedding. Vector dimensions', edim            
            
            return external_embedding, edim, noextrn, extrnd, elookup
    
        return None,None,None,None,None



    def __evaluate(self, c, train):
              
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
            #In the test phase we already pick the most likely transition/dependency 
            #In Covington we have to select the best valid
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
        evec = self.elookup[1] if self.external_embedding is not None else None
        paddingWordVec = self.wlookup[1]
        paddingPosVec = self.plookup[1] if self.pdims > 0 else None

        paddingVec = tanh(self.word2lstm.expr() * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias.expr() )
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])


    def getWordEmbeddings(self, sentence, train):
        
 #       print [(word.form, word.cpos) for word in sentence], train
        for root in sentence:
            print "Llega 1"
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))
            print "Llega 1.25"
            sys.stdout.flush()
            root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0]
            print "Llega 1.5"
            root.cposvec = self.plookup[int(self.cpos.get(root.cpos,0))] if self.pdims > 0 else None

            print "Llega 2"
            #For word embeddings
            if self.external_embedding is not None:
                #if not dropFlag and random.random() < 0.5:
                #    root.evec = self.elookup[0]
                if root.form in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.norm]]
                else:
                    root.evec = self.elookup[0]
            else:
                root.evec = None
            
            print "Llega 3"
            #For postag embeddings
            if self.cpos_external_embedding is not None:
                if root.cpos in self.cpos_external_embedding:
                    root.cposevec = self.cpos_elookup[self.cpos_extrnd[root.cpos]]
                elif root.norm in self.external_embedding:
                    root.cposevec = self.cpos_elookup[self.cpos_extrnd[root.cpos]]
                else:
                    root.cposevec = self.cpos_elookup[0]
            else:
                root.cposevec = None
                
            print "Llega 4"
            root.ivec = concatenate(filter(None, [root.wordvec, root.cposvec, root.evec, root.cposevec]))
            

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
        
     #   print ("Entra predict")
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
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
                        
                        sentence[b].pred_parent_id = sentence[l1].id # l1 should be the same as sentence[l1].id
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
        print "Llega 1"
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
            shuffledData = list(read_conll(conllFP, True))
            
     #       print len(shuffledData)
            random.shuffle(shuffledData)
            
            #random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
            #    print iSentence, len(sentence), sentence[0], type(sentence[0]), sentence[0].id, sentence[0].form,sentence[0].parent_id,sentence[0].relation
            #    print iSentence, len(sentence), sentence[1], type(sentence[1]), sentence[1].id, sentence[1].form,sentence[1].parent_id,sentence[1].relation
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start                  
#                     print 'Predicted arcs', arcs
#                     print 'Gold arcs', gold_arcs       
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                print "Llega 2"
                self.getWordEmbeddings(sentence, True)
                print "Llega 3"
                
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
                loss_c = self._loss(c,gold_arcs)
            
                for word in sentence:
                    word.lstms = [word.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0
#                 print ("l1 before starting",sentence[l1].id, sentence[l1].form, sentence[l1].parent_id, sentence[l1].relation)
#                 print ("b before starting",sentence[b].id, sentence[b].form, sentence[b].parent_id, sentence[b].relation)
                while not self._is_final_state(b,sentence):

                    costs = [None,None,None,None]
                    transition_scores = self.__evaluate(c, True)
#                     print ("Gold arcs", gold_arcs)
               #     print ("transition_scores",transition_scores)
               #     raw_input()
#                     print ("costs",costs)
                    for t,_ in enumerate(transition_scores):
                        
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
                            loss_new_c = self._loss(new_c,gold_arcs)
                        
                            cost = loss_new_c - loss_c
                            costs[t] = float(cost)

#                     print ("updated costs", costs)
                #    print ("aaa",[s for s in chain(*transition_scores)]) 
                    #Valid transitions left and right arcs with cost = 0 & shift or no-arc
#                    print ("list(chain*transitions_scores)",list(chain(*transition_scores)))
                 
                    #TODO: This can be improved
                    valid_transitions = [s for s in chain(*transition_scores) if costs[s[1]] == 0 and (s[1] in [self.SHIFT,self.NO_ARC] 
                                                                                                          or ((s[1] == self.LEFT_ARC and s[0] == sentence[l1].relation) 
                                                                                                              or (s[1] == self.RIGHT_ARC and s[0] == sentence[b].relation))  )]
                    
                    
      #              print ("Estimated costs", costs)
      #              print ("valid_transitions", valid_transitions)

                    best_valid = max(valid_transitions, key=itemgetter(2))
                       
                    
     #               print ("best_valid", best_valid)


                    wrong_transitions = [s for s in chain(*transition_scores) if costs[s[1]] is not None and ( (costs[s[1]] != 0) or (s[1] in [self.LEFT_ARC,self.RIGHT_ARC] 
                                                                                                          and ((s[1] == self.LEFT_ARC and s[0] != sentence[l1].relation) 
                                                                                                              or (s[1] == self.RIGHT_ARC and s[0] != sentence[b].relation))) ) ]
                    
       #             print ("wrong transitions", wrong_transitions)                     

#                    best = best_valid
                    if wrong_transitions != []:
                        best_wrong = max(wrong_transitions, key=itemgetter(2))    
          #              print ("best wrong", best_wrong)
#     
#                         
                        best = best_valid if ( (not self.oracle) or (best_valid[2] - best_wrong[2] > 1.0) 
                                              or (best_valid[2] > best_wrong[2] and random.random() > 0.1) ) else best_wrong
                    else:
                        best = best_valid 

#                     print ("best", best)
#                     print ("best", best)

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
                    loss_c = self._loss(c,gold_arcs)
                 
              #  print "Entra", iSentence

                if len(errs) > 50: # or True:
                    #eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
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
        # while len(buf) > 0 or len(stack) > 1 :
        return b >= len(sentence)

    def _is_valid_left_arc(self,c):

        aux = set(c.A)
        aux.add((c.b,c.l1))

        l1_has_head = self._y_has_head(c.A, c.b, c.l1)
        
#         print ("_is_valid_left_arc l1", l1)
#         print ("_is_valid_left_arc l1_has_head", l1_has_head)
#         print ("_is_valid_left_arc l1>0", l1>0)
#         print ("self._count_cycles(aux)", self._count_cycles(aux), self._count_cycles(aux) == 0)
        
        return (c.l1 > 0 and not l1_has_head
                and self._count_cycles(aux) == 0)

    def _is_valid_right_arc(self,c):
        
        b_has_head = self._y_has_head(c.A, c.l1, c.b)
        
        
#         print ("is_valid_right_arc",l1,b, sentence[l1],sentence[b])
#         print ("c.A", c.A)
        aux = set(c.A)
        aux.add((c.l1,c.b))
        
#         print ("_is_valid_right_arc l1,b", l1,b)
#         print ("_is_valid_right_arc b_has_head", b_has_head)
#         print ("self._count_cycles(aux) == 0",self._count_cycles(aux),self._count_cycles(aux) == 0)
        
#         print ("sentence[b]",sentence[b].parent_id, sentence[b].pred_parent_id)
#         print ("sentence[l1]",sentence[l1].parent_id, sentence[l1].pred_parent_id)
        
        


#         print ("_is_valid_right_arc l1", l1)
#         print ("_is_valid_right_arc l1_has_head", b_has_head)
#         print ("self._count_cycles(aux)", self._count_cycles(aux), self._count_cycles(aux) == 0)


        return ((not b_has_head) and self._count_cycles(aux) == 0)
        
        
    """
    Gomez-Rodriguez & Fernandez-Gonzalez: 
    An Efficiente Dynamic Oracle for Unrestricted Non-Projective Parsing  (ACL,2015)
    Algorithm 1
    """
    def _loss(self, c, gold_arcs):
        
     #   print ("State in loss",str(c))
        U = set([]) #set of unreachable nodes
        non_built_arcs = gold_arcs.difference(c.A)
        
        
        i = c.l1
        j = c.b
              
        for x,y in non_built_arcs: 
            left = min(x,y)  #O(n)
            right = max(x,y) #O(n)
            if (j > right or (j==right and i < left) or self._y_has_head(c.A,x,y)
                or self._weakly_connected(c.A, x, y,c, gold_arcs)):
            #    print i,j,"|",x,y, j > right,(j==right and i < left), self._y_has_head(c.A,x,y),self._weakly_connected(c.A, x, y,c, gold_arcs)
                U.add((x,y))
        
        I = gold_arcs.difference(U)

        #print len(U), U, non_built_arcs, c.A.union(I), self._count_cycles( c.A.union(I))
        return len(U) + self._count_cycles( c.A.union(I))
    
    
    #TODO: This can be done much better
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
        

    
    
    #O(n)
    def _y_has_head(self,A,x,y):
        
        for z,y_prime in A:
            if y_prime == y and z != x:
                return True
        return False
     
