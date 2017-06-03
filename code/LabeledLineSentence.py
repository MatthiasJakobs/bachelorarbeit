from random import shuffle
from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath) as f:
            for n, line in enumerate(f):
                yield LabeledSentence(line.split(), ['SENT_%s' % n])

    def toArray(self):
        self.sentences = []
        with open(self.filepath) as f:
            for n, line in enumerate(f):
                self.sentences.append(LabeledSentence(line.split(), ['SENT_%s' % n]))
        return(self.sentences)

    def permutation(self):
        shuffle(self.sentences)
        return(self.sentences)
