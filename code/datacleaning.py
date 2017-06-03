import json
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import nltk

def filterWordlist(wordlist, corpus):
    newline = []
    withoutwords = []
    for line in corpus:
        for word in line:
            if word not in wordlist:
                newline.append(word)

        withoutwords.append(newline)
        newline = []

    return withoutwords

def cleanString(input, lang="en", removeStopWords=True, useStemmer=True):
    split = input.lower().split()

    if removeStopWords:
        stop_words = get_stop_words(lang)
        # also remove single digits because they are not really useful
        stop_words.extend("a b c d e f g h j i k l m n o p q r s t u v w x y z".split())

    # We don't use retweets in our data
    if "rt" in split:
        return []

    clean = []
    stemmer = PorterStemmer()
    for word in split:
        if not re.search(r'(http|@|\#)',word):
            nopunc = re.sub(r'(\?|\.|\,|\:|\;|\-|\—|\‘|\_|\!|\"|\'|\’|\”|\“)', "", word)
            if removeStopWords:
                if removeStopWords and nopunc not in stop_words:
                    cleaned = nopunc
                    if lang == 'en' and useStemmer:
                        stemmed = stemmer.stem(nopunc)
                        cleaned = stemmed
                    clean.append(cleaned)
            else:
                if useStemmer:
                    stemmer = PorterStemmer()
                    stemmed = stemmer.stem(nopunc)
                    nopunc = stemmed
                clean.append(nopunc)
                
    return clean

def createHalfcleanedFile(path):
    newPath = path + ".halfclean"
    beforeCounter = 0
    afterCounter = 0
    skipcounter = 0
    input = cleanRawTweetsFromFile(path, removeStopWords=False, stemm=False)
    for line in input:
        with open(newPath, "a") as output:
            output.write(" ".join(line))
            output.write("\n")

def createW2VnoStop(path):
    newPath = path + ".d2vclean1"
    beforeCounter = 0
    afterCounter = 0
    skipcounter = 0
    stemmer = PorterStemmer()
    input = cleanRawTweetsFromFile(path, removeStopWords=False, stemm=True)
    for line in input:
        with open(newPath, "a") as output:
            output.write(" ".join(line))
            output.write("\n")

def createW2VnoStem(path):
    newPath = path + ".d2vclean2"
    beforeCounter = 0
    afterCounter = 0
    skipcounter = 0
    input = cleanRawTweetsFromFile(path, removeStopWords=True, stemm=False)
    for line in input:
        with open(newPath, "a") as output:
            output.write(" ".join(line))
            output.write("\n")

def createCleanedFile(path):
    newPath = path + ".clean"
    beforeCounter = 0
    afterCounter = 0
    skipcounter = 0
    stemmer = PorterStemmer()
    input = cleanRawTweetsFromFile(path)
    for line in input:
        with open(newPath, "a") as output:
            output.write(" ".join(line))
            output.write("\n")


def cleanRawTweetsFromFile(path, lang="en", stemm=True, removeStopWords=True, verbose=False, minsize=5):
    # Removes punctuation, retweets, @-mentions, links and hashtags
    # Lowercases and returns list of list of words
    # Maybe remove stopwords, stemms and filters 'useless' tweets

    print("Pfad: " + path)
    i = 0
    skipped = 0
    clean = []
    with open(path) as f:
        for line in f:
            try:
                clean.append(cleanString(json.loads(line)["text"], removeStopWords=removeStopWords, useStemmer=stemm))
                #print('added new: ' + str(i))
                i = i+1
            except:
                skipped += 1
                continue

    print("Skipped: " + str(skipped))

    # remove short ('meaningless') tweets
    toReturn = [line for line in clean if len(line) >= minsize]
    print("After Cleanup: " + str(len(toReturn)))
    return toReturn
