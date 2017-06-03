from utilities import LOG
from utilities import removeIfExists
import constants as CONST
import os
import numpy as np
from math import isnan,log
from random import shuffle
from gensim import corpora
from gensim import models
from collections import Counter
from LabeledLineSentence import *
from gensim.matutils import sparse2full
#from glove import Glove, Corpus

class AbstractModel(object):
    def build(self):
        return NotImplemented

    def computeDocumentVector(self):
        return NotImplemented

    def infer(self, input):
        return NotImplemented

class TFIDFModel(AbstractModel):
    def __init__(self, name, verbose=True):
        self.verbose = verbose
        self.name = name
        self.tweetPath = "cleaned/" + name + ".clean"
        self.docvecpath = "testing/" + name + "_tfidf_docvec.npy"


    def getVocabSize(self):
        return len(self.dictionary.items())

    def build(self):
        tweets = [line.split() for line in open(self.tweetPath)]

        if not os.path.exists("testing/" + self.name + "_.dict"):
            # map the words of the corpus to an id
            if self.verbose: LOG("Recreate dictionary...")
            self.dictionary = corpora.Dictionary(line for line in tweets)
            self.dictionary.save("testing/" + self.name + "_.dict")
        else:
            if self.verbose: LOG("Loading dictionary...")
            self.dictionary = corpora.Dictionary.load("testing/" + self.name + "_.dict")

        if not os.path.exists("testing/" + self.name + "_.corpus"):
            # vectors of type (wordID, freq in line)
            # in dense form: with words of corpus are in line (1) or not (0)
            if self.verbose: LOG("Recreate corpus...")
            self.corpus = [self.dictionary.doc2bow(line) for line in tweets]
            corpora.MmCorpus.serialize("testing/" + self.name + "_.corpus", self.corpus)
        else:
            if self.verbose: LOG("Loading corpus...")
            self.corpus = corpora.MmCorpus("testing/" + self.name + "_.corpus")

        if not os.path.exists("testing/" + self.name + "_.tfidf"):
            if self.verbose: LOG("Recreate TF-IDF model...")
            self.tfidf = models.TfidfModel(self.corpus)
            self.tfidf.save("testing/" + self.name + "_.tfidf")
            self.computeDocumentVector()
        else:
            if self.verbose: LOG("Loading TF-IDF model...")
            self.tfidf = models.TfidfModel.load("testing/" + self.name + "_.tfidf")
            self.documentVector = list(np.load(self.docvecpath))



    def computeDocumentVector(self):
        removeIfExists(self.docvecpath)
        n = len(self.dictionary.items()) # Number of unique words in dictionary
        n2 = len(self.corpus)

        result = [0]*n
        for i in range(0,n2):
            a = sparse2full(self.tfidf[self.corpus[i]], n)
            result = result + a

        self.documentVector = list(map(lambda x: x/n, result))
        np.save(self.docvecpath, np.array(self.documentVector))

        #self.documentVector = result

    def infer(self, vector):
        return list(sparse2full(self.tfidf[self.dictionary.doc2bow(vector)], len(self.dictionary.items())))

class CombinedModel(AbstractModel):
    def __init__(self, name, verbose=True):
        self.name = name
        self.tweetpath = "cleaned/" + name + ".clean"
        self.docvecpath = "testing/" + name + "_comb_docvec.npy"

    def build(self):
        self.w2vmodel = W2VModel(self.name, sg=0)
        self.w2vmodel.build()
        self.tfidfmodel = TFIDFModel(self.name)
        self.tfidfmodel.build()
        self.computeDocumentVector()

    def computeDocumentVector(self):
        LOG("Recrate document vector")
        removeIfExists(self.docvecpath)
        sentences = [line.split() for line in open(self.tweetpath)]
        docvec = np.array([0.0]*750)
        for sentence in sentences:
            infered = self.infer(sentence)
            docvec = docvec + self.infer(sentence) / 750
        
        np.save(self.docvecpath, docvec)
        self.documentVector = list(docvec)

    def infer(self, sentence):
        sentence_vec = np.array([0.0]*750)
        tfidfvector = self.tfidfmodel.infer(sentence)
        i = 0
        for word in sentence:
            w2vvector = self.w2vmodel.infer_word(word)
            #print(w2vvector)
            sentence_vec += w2vvector * tfidfvector[i]
            i += 1
        return sentence_vec

class W2VModel(AbstractModel):
    def __init__(self, name, sg=0, window=5, N=750, verbose=True, useNS=False, useGoogle=False):
        self.verbose = verbose
        self.featureVectorSize = N
        self.name = name
        #self.tweetPath = "cleaned/" + name + ".halfclean"
        if sg == 1:
            self.window = 10
            self.tweetPath = "cleaned/" + name + ".halfclean"
        else:
            self.window = 5
            self.tweetPath = "cleaned/" + name + ".clean"
        self.sg = sg
        self.window = window
        self.google = useGoogle
        if useNS:
            self.ns = 5
        else:
            self.ns = 0
        self.modelpath = "testing/" + self.name + "_.word2vec_" + str(self.sg) + "_" + str(self.featureVectorSize)
        self.docvecpath = self.modelpath + "_docvec.npy"

    def overwriteTraindata(self, clean=True, path=None):
        if path != None:
            self.tweetPath = path
        else:
            if clean:
                self.tweetPath = "cleaned/" + self.name + ".clean"
            else:
                self.tweetPath = "cleaned/" + self.name + ".halfclean"

    def build(self):
        self.w2vtweets = [line.split() for line in open(self.tweetPath)]
        if not os.path.exists(self.modelpath):
            if self.verbose: LOG("Recreate word2vec model...")
            self.w2vmodel = models.Word2Vec(self.w2vtweets, sg=self.sg, negative=self.ns, window=self.window, size=self.featureVectorSize)
            if self.verbose: LOG("Start training word2vec model...")
            for e in range(30):
                self.w2vmodel.train(self.w2vtweets)
                self.w2vmodel.alpha *= 0.99
                self.w2vmodel.min_alpha = self.w2vmodel.alpha
                if self.verbose: LOG("Iteration " + str(e+1) + " completed")
            self.w2vmodel.save(self.modelpath)
            self.computeDocumentVector()
        else:
            if self.verbose: LOG("Loading word2vec model...")
            self.w2vmodel = models.Word2Vec.load(self.modelpath)
            if not os.path.exists(self.docvecpath):
             self.computeDocumentVector()
            else:
                LOG("Loading document vector")
                self.documentVector = list(np.load(self.docvecpath))

    def infer_word(self, word):
        try:
            return self.w2vmodel[word]
        except KeyError:
            return np.array([0.0]*self.featureVectorSize)

    def infer(self, sentence):
        sent_vec = np.array([0.0]*self.featureVectorSize)
        for word in sentence:
            sent_vec = sent_vec + self.infer_word(word) / self.featureVectorSize
        return sent_vec

    def computeDocumentVector(self):
        LOG("Recrate document vector")
        removeIfExists(self.docvecpath)
        sentences = self.w2vtweets
        docvec = np.array([0.0]*self.featureVectorSize)
        for sentence in sentences:
            infered = self.infer(sentence)
            docvec = docvec + self.infer(sentence) / self.featureVectorSize
        
        np.save(self.docvecpath, docvec)
        self.documentVector = list(docvec)

class LDAModel(AbstractModel):
    def __init__(self, name, N=750, verbose=True):
        self.name = name
        self.verbose = verbose
        self.tweetPath = "cleaned/" + name + ".clean"
        self.featureLength = N
        self.docvecpath = "testing/lda_" + name + "_docvec.npy"

    def build(self):
        self.tweets = [line.split() for line in open(self.tweetPath)]
        if not os.path.exists("testing/" + self.name + "_.ldadict_" + str(self.featureLength)):
            # map the words of the corpus to an id
            if self.verbose: LOG("Recreate LDA dictionary...")
            self.dictionary = corpora.Dictionary(line for line in self.tweets)
            self.dictionary.save("testing/" + self.name + "_.ldadict_" + str(self.featureLength))
        else:
            if self.verbose: LOG("Loading LDA dictionary...")
            self.dictionary = corpora.Dictionary.load("testing/" + self.name + "_.ldadict_" + str(self.featureLength))

        if not os.path.exists("testing/" + self.name + "_.ldacorpus_" + str(self.featureLength)):
            # vectors of type (wordID, freq in line)
            # in dense form: with words of corpus are in line (1) or not (0)
            if self.verbose: LOG("Recreate LDA corpus...")
            self.corpus = [self.dictionary.doc2bow(line) for line in self.tweets]
            corpora.MmCorpus.serialize("testing/" + self.name + "_.ldacorpus_" + str(self.featureLength), self.corpus)
        else:
            if self.verbose: LOG("Loading LDA corpus...")
            self.corpus = corpora.MmCorpus("testing/" + self.name + "_.ldacorpus_" + str(self.featureLength))

        if not os.path.exists("testing/" + self.name + "_.lda_" + str(self.featureLength)):
            if self.verbose: LOG("Recreate LDA model...")
            self.ldamodel = models.LdaModel(self.corpus, num_topics=self.featureLength)
            self.ldamodel.save("testing/" + self.name + "_.lda_" + str(self.featureLength))
            self.computeDocumentVector()
        else:
            if self.verbose: LOG("Loading LDA model...")
            self.ldamodel = models.LdaModel.load("testing/" + self.name + "_.lda_" + str(self.featureLength))
            self.documentVector = list(np.load(self.docvecpath))

    def computeDocumentVector(self):
        LOG("Start rebuilding document vector")
        removeIfExists(self.docvecpath)
        documentVector = np.array([0]*self.featureLength)
        for document in self.tweets:
            vec = self.infer(document)
            documentVector = documentVector + vec

        self.documentVector = list(documentVector/len(self.tweets))
        np.save(self.docvecpath, documentVector)

    def infer(self,sentence):
        vec = [0]*self.featureLength
        for (tid,tvalue) in self.ldamodel.get_document_topics(self.dictionary.doc2bow(sentence)):
            vec[tid] = tvalue
        return np.array(vec)
