from datetime import datetime
import datacleaning
import numpy as np
import scipy
import constants as CONST
import webscraper
from os.path import exists
from os import remove
from random import shuffle
from math import ceil
import json

def removeIfExists(path):
    if(exists(path)):
        LOG("removing " + path)
        remove(path)

def LOG(text):
    now = datetime.now()
    print('{:%H:%M:%S}'.format(datetime.now()) + " " + text)

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def flatten(l):
    return [item for sublist in l for item in sublist]


def getLabeledTestdata(name, verbose=True, fullclean=True, onlystemming=False, onlystop=False, identicalNegatives=False, randomNegatives=False):
    correctPath = "testing/" + name + "_correct"
    incorrectPath = "testing/" + name + "_incorrect"
    correctLinkFile = "../corpora/articles/" + name + ".txt"
    if name == "muslimban":
        incNumber = "1"
    elif name == "flynn":
        incNumber = "2"
    elif name == "jointaddress":
        incNumber = "3"
    elif name == "killthebill":
        incNumber = "4"
    else:
        incNumber = "5"
    if identicalNegatives:
        incNumber = "1"
    incorrectLinkFile = "../corpora/articles/misc" + incNumber + ".txt"
    if not exists(correctPath):
        if verbose: LOG("Downloading testing data for 'correct'...")
        correct_docs = webscraper.useLinkFile(correctLinkFile)
        with open(correctPath, "w+") as f:
            for doc in correct_docs:
                f.write(json.dumps(doc))
                f.write('\n')
    else:
        if verbose: LOG("Get testing data of 'correct' file...")
        correct_docs = []
        with open(correctPath) as f:
            for line in f.readlines():
                correct_docs.append(json.loads(line))

    if not randomNegatives:
        if verbose: LOG("Get testing data of 'incorrect' file...")
        incorrect_docs = []
        with open("../corpora/articles/misc" + incNumber + "_saved.txt") as f:
            for line in f.readlines():
                incorrect_docs.append(json.loads(line))
    else:
        paths = ["../corpora/articles/misc1_saved.txt",
                "../corpora/articles/misc2_saved.txt",
                "../corpora/articles/misc3_saved.txt",
                "../corpora/articles/misc4_saved.txt",
                "../corpora/articles/misc5_saved.txt"]

        negativeArticles = []
        for path in paths:
            with open(path) as f:
                negativeArticles.append(f.readlines())
        negativeArticles = sum(negativeArticles, [])
        shuffle(negativeArticles)
        incorrect_docs = []
        for article in negativeArticles[:45]:
            incorrect_docs.append(json.loads(article))


    testdocs = []

    for doc in correct_docs:
        label = " ".join(doc["headline"])
        fulltext = doc["headline"]
        fulltext.extend(doc["body"])
        cleaned = datacleaning.cleanString(" ".join(fulltext))
        halfcleaned = datacleaning.cleanString(" ".join(fulltext), useStemmer=False, removeStopWords=False)
        stopclean = datacleaning.cleanString(" ".join(fulltext), useStemmer=False, removeStopWords=True)
        stemmclean = datacleaning.cleanString(" ".join(fulltext), useStemmer=True, removeStopWords=False)
     
        if fullclean:
            testdocs.append((label, cleaned, True))
        elif onlystemming:
            testdocs.append((label, stemmclean, True))
        elif onlystop:
            testdocs.append((label, stopclean, True))
        else:
            testdocs.append((label, halfcleaned, True))

    for doc in incorrect_docs:
        label = " ".join(doc["headline"])
        fulltext = doc["headline"]
        fulltext.extend(doc["body"])
        cleaned = datacleaning.cleanString(" ".join(fulltext))
        halfcleaned = datacleaning.cleanString(" ".join(fulltext), useStemmer=False, removeStopWords=False)
        stopclean = datacleaning.cleanString(" ".join(fulltext), useStemmer=False, removeStopWords=True)
        stemmclean = datacleaning.cleanString(" ".join(fulltext), useStemmer=True, removeStopWords=False)
     
        if fullclean:
            testdocs.append((label, cleaned, False))
        elif onlystemming:
            testdocs.append((label, stemmclean, False))
        elif onlystop:
            testdocs.append((label, stopclean, False))
        else:
            testdocs.append((label, halfcleaned, False))

    return testdocs

def calcAllMeanLenghts():
    names = ['muslimban', 'flynn', 'killthebill', 'jointaddress', 'syriastrikes']
    for name in names:
        calcMeanTweetLength(name)

def calcMeanTweetLength(name):
    path = "cleaned/" + name + ".clean"
    mean = 0
    with open(path) as f:
        tweets = f.readlines()
    print(str(len(tweets)))

    for tweet in tweets:
        length = len(tweet.split())
        mean += length / len(tweets)

    print(name + ", " + str(mean))
