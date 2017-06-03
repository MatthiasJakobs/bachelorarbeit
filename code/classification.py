import numpy as np
import utilities
import models
import os
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer


def cosineSimilarityScores(testdocs, model, threshold):
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    scores = []

    for (label, cleaned, indicator) in testdocs:

        infered = model.infer(cleaned)

        score = cosine_similarity(model.documentVector, infered)

        if score > threshold:
            scores.append((label[0:60], '{0:.5f}'.format(score), "IN"))
            if(indicator == True):
                tp = tp + 1
            else:
                fn = fn + 1

        else:
            scores.append((label[0:60], '{0:.5f}'.format(score), "NOT IN"))
            if(indicator == False):
                tn = tn + 1
            else:
                fp = fp + 1

    rec = getRecall(tp, fn)
    prec = getPrecision(tp, fp)
    return (scores, (rec,prec))

def getFitnessWithRandomTestdata(name, verbose=True, iterations=50):
    tfidfModel = models.TFIDFModel(name, verbose=verbose)
    tfidfModel.build()
    tfidfModel.computeDocumentVector()

    word2vecModel1 = models.W2VModel(name, verbose=verbose)
    word2vecModel1.build()
    word2vecModel1.computeDocumentVector()

    word2vecModel2 = models.W2VModel(name, sg=1, window=10, verbose=verbose)
    word2vecModel2.build()
    word2vecModel2.computeDocumentVector()

    f1 = []
    f2 = []
    f3 = []
    indices = []

    i = 0
    while i < iterations:
        #testdocshalfclean = utilities.getLabeledTestdata(name, verbose=verbose, fullclean=False, randomNegatives=True)
        #print("%.2f" % a)
        indices.append(i)
        (tfidfScores, (tfidfrec, tfidfprec)) = cosineSimilarityScores(utilities.getLabeledTestdata(name, verbose=False, randomNegatives=True) , tfidfModel, 0.25)
        f1.append(f1Measure(tfidfrec, tfidfprec))
        (w2vScores1, (w2vrec1, w2vprec1)) = cosineSimilarityScores(utilities.getLabeledTestdata(name, verbose=False, randomNegatives=True), word2vecModel1, -0.1)
        f2.append(f1Measure(w2vrec1, w2vprec1))
        (w2vScores2, (w2vrec2, w2vprec2)) = cosineSimilarityScores(utilities.getLabeledTestdata(name, verbose=False, randomNegatives=True), word2vecModel2, 0.85)
        f3.append(f1Measure(w2vrec2, w2vprec2))
        if verbose: print("Iteration " + str(i+1) + " done")
        if os.path.exists("dump.txt"):
            os.remove("dump.txt")
        with open("dump.txt", "a") as f:
            f.write(str(f1Measure(tfidfrec, tfidfprec)) + "\t" + str(f1Measure(w2vrec1, w2vprec1)) + "\t" + str(f1Measure(w2vrec2, w2vrec2)))
            f.write("\n")
        i += 1

    return (indices, f1, f2, f3)  


def ClusterKMeans(model, testdocs):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    
    vectors = []
    sim_min = 1
    sim_max = 0

    full_articles = []

    n_positives = 0
    n_negatives = 0

    for (label, cleaned, indicator) in testdocs:
        similarity = cosine_similarity(model.documentVector, model.infer(cleaned))
        if similarity > sim_max:
            sim_max = similarity
        if similarity < sim_min:
            sim_min = similarity
        #vectors.append([0, similarity])
        vectors.append([similarity])
        if indicator:
            n_positives += 1
        else:
            n_negatives += 1
        
        full_articles.append(label)

    #kmeans = KMeans(n_clusters=2, random_state=0, n_init=1, init=np.array([[0, sim_min], [0, sim_max]])).fit(vectors)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=1, algorithm="full", init=np.array([[sim_min], [sim_max]])).fit(vectors)
    #kmeans = KMeans(n_clusters=2, random_state=0, n_init=100).fit(vectors)

    clusters = kmeans.labels_
    positives = clusters[:n_positives-1]
    negatives = clusters[-n_negatives-1:]
    #docveccluster = kmeans.predict([[0,1]])
    docveccluster = kmeans.predict([[1]])

    positive_articles = []
    negative_articles = []

    for i in range(len(clusters)):
        if clusters[i] == docveccluster:
            positive_articles.append(full_articles[i])
        else:
            negative_articles.append(full_articles[i])
            

    for decision in positives:
        if decision == docveccluster[0]:
            tp = tp + 1
        else:
            fn = fn + 1
    
    for decision in negatives:
        if decision != docveccluster[0]:
            tn = tn + 1
        else:
            fp = fp + 1

    rec = getRecall(tp, fn)
    prec = getPrecision(tp, fp)
    return (f1Measure(rec,prec), positive_articles, negative_articles)



def testHashtag(name, useTFIDF=True, useD2V=False, useW2V=True, useLDA=True,verbose=True, identicalNegatives=True):
    testdocsclean = utilities.getLabeledTestdata(name, verbose=verbose, identicalNegatives=identicalNegatives)
    testdocshalfclean = utilities.getLabeledTestdata(name, verbose=verbose, fullclean=False, identicalNegatives=identicalNegatives)
    print("-" * 60)
    print("\t\t\t" + name)
    print("-" * 60)

    tfidfModel = models.TFIDFModel(name, verbose=verbose)
    tfidfModel.build()
    tfidfModel.computeDocumentVector()

    if useTFIDF:

        (tfidfScores, (tfidfrec, tfidfprec)) = cosineSimilarityScores(testdocsclean, tfidfModel, 0.25)

        print("-" * 60)
        print("\t\t tfidf")
        print("-" * 60)

        if verbose:
            for line in tfidfScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(tfidfrec, tfidfprec)))

    if useD2V:
        d2vModel = models.D2VModel(name, bow=True, verbose=verbose)
        d2vModel.build()
        d2vModel.computeDocumentVector()
        (d2vScores, (d2vrec, d2vprec)) = cosineSimilarityScores(testdocshalfclean, d2vModel, 0.2)


        print("-" * 60)
        print("\t\t doc2vec - DBOW")
        print("-" * 60)

        if verbose:
            for line in d2vScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(d2vrec, d2vprec)))

        d2vModel = models.D2VModel(name, bow=False, verbose=verbose)
        d2vModel.build()
        d2vModel.computeDocumentVector()
        (d2vScores, (d2vrec, d2vprec)) = cosineSimilarityScores(testdocshalfclean, d2vModel, 0.15)


        print("-" * 60)
        print("\t\t doc2vec - DM")
        print("-" * 60)

        if verbose:
            for line in d2vScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(d2vrec, d2vprec)))


    if useW2V:
        word2vecModel = models.W2VModel(name, verbose=verbose)
        word2vecModel.build()
        word2vecModel.computeDocumentVector()

        (w2vScores, (w2vrec, w2vprec)) = cosineSimilarityScores(testdocsclean, word2vecModel, -0.1)
        print("-" * 60)
        print("\t\t word2vec - cbow")
        print("-" * 60)

        if verbose:
            for line in w2vScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(w2vrec, w2vprec)))

        word2vecModel = models.W2VModel(name, sg=1, window=10, verbose=verbose)
        word2vecModel.build()
        word2vecModel.computeDocumentVector()

        (w2vScores, (w2vrec, w2vprec)) = cosineSimilarityScores(testdocsclean, word2vecModel, 0.85)
        print("-" * 60)
        print("\t\t word2vec - skipgram")
        print("-" * 60)

        if verbose:
            for line in w2vScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(w2vrec, w2vprec)))

    if useLDA:
        ldamodel = models.LDAModel(name)
        ldamodel.build()
        ldamodel.computeDocumentVector()

        (ldaScores, (ldarec, ldaprec)) = cosineSimilarityScores(testdocsclean, ldamodel, 0.85)
        print("-" * 60)
        print("\t\t LDA")
        print("-" * 60)

        if verbose:
            for line in ldaScores:
                print(line)

        print("-" * 60)

        print("f: " + str(f1Measure(ldarec, ldaprec)))


def getPrecision(tp, fp):
    if(tp + fp > 0):
        return tp / (tp + fp)
    elif tp == 0:
        return 0.001
    else:
        return tp / 1

def getRecall(tp, fn):
    if(tp + fn > 0):
        return tp / (tp + fn)
    elif tp == 0:
        return 0.001
    else:
        return tp / 1

def cosine_similarity(x,y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

def f1Measure(rec,prec):
    if rec + prec == 0:
        return 2 * rec * prec / 0.001
    elif rec * prec == 0:
        return 0.001
    else:
        return 2*((rec * prec) / (rec + prec))
