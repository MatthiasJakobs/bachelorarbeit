import plotting
import utilities
import classification
import matplotlib.pyplot as plt
import numpy as np
import models
from os.path import exists
from os import remove

def word2vecFeatureSize(cbow):
    vectorSizes =   [50, 100, 250, 500, 750, 1000, 2000]
    colors =        ["r","b", "g", "c", "m", "y",  "k"]
    i = 0
    if cbow:
        testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=False, identicalNegatives=True)
        sg = 0
    else:
        testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=True, identicalNegatives=True)
        sg = 1

    for size in vectorSizes:
        utilities.removeIfExists("testing/syriastrikes_.word2vec_" + str(sg) + "_" + str(size))
        
    
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0.4,1,0.05))

    for size in vectorSizes:
        model = models.W2VModel("syriastrikes", sg=sg, useNS=True, N=size)
        model.build()
        model.computeDocumentVector()

        f = []
        indices = []

        if cbow:
            under = 0
            upper = 0.6
        else:
            under = 0.7
            upper = 1

        thres = under
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            #print(scores)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)
            #print(str(2 * ((acc * prec) / (acc + prec))))

            #f.append(acc * prec)
            thres += 0.001
        #ind = np.array(len(indices))
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(list(map(lambda x: str(x), vectorSizes)))
    #plt.show()
    plt.savefig("../vorarbeit/experimente/bilder/newPlots/w2vVector_" + str(sg) + "_Final.png")


def LDAFeatureSize():
    vectorSizes =   [50, 100, 250, 500, 750, 1000, 2000]
    colors =        ["r","b", "g", "c", "m", "y",  "k"]
    i = 0
    testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=True, identicalNegatives=True)
    
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))

    for size in vectorSizes:
        model = models.LDAModel("syriastrikes", N=size)
        model.build()
        model.computeDocumentVector()

        f = []
        indices = []

        under = 0
        upper = 1
        thres = under
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            #print(scores)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)
            #print(str(2 * ((acc * prec) / (acc + prec))))

            #f.append(acc * prec)
            thres += 0.001

        #ind = np.array(len(indices))
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(list(map(lambda x: str(x), vectorSizes)))
    plt.show()

def randomTestdata():
    model1 = models.TFIDFModel("syriastrikes")
    model1.build()
    model2 = models.W2VModel("syriastrikes", sg=0)
    model2.build()
    model3 = models.W2VModel("syriastrikes", sg=1)
    model3.build()
    model4 = models.LDAModel("syriastrikes")
    model4.build()

    iterations = 35
    count = 1

    #testmodels = [model1, model2, model3, model4]
    testmodels = [model1, model2, model3, model4]
    testclean = [True, False, True, True]

    # utilities.removeIfExists("negative_dump_data")
    # with open("negative_dump_data", "a") as fil:
    #     fil.write("tfidf" + "\t" + "cbow" + "\t" + "skip" + "\t" + "lda")
    #     fil.write("\n")
    # fil.close()

    for j in range(iterations):
        fi = []
        for i in range(len(testmodels)):
            testdata = utilities.getLabeledTestdata("syriastrikes", verbose=False, fullclean=testclean[i], randomNegatives=True)
            (f1score, positives, negatives) = classification.ClusterKMeans(testmodels[i], testdata)
            fi.append(f1score)
            utilities.LOG("Progress: " + str(count/(iterations*4)))
            count += 1
        with open("negative_dump_data", "a") as fil:
            fil.write(str(fi[0]) + "\t" + str(fi[1]) + "\t" + str(fi[2]) + "\t" + str(fi[3]))
            fil.write("\n")
        fil.close()

def differentTrainAndTest(cbow):
    if cbow:
        sg = 0
        window = 5
    else:
        sg = 1
        window = 10

    testdata = [True, False]
    traindata = [True, False]
    colors = ["r",           "g",               "b",               "m"]
    labels = ["Clean/Clean", "Clean/Halfclean", "Halfclean/Clean", "Halfclean/Halfclean"]
    
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))

    i = 0
    exports = []
    for train in traindata:
        path = "testing/syriastrikes_.word2vec_" + str(sg) + "_" + str(750)
        if(exists(path)):
            print("removing " + path)
            remove(path)
        for test in testdata:
            testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=test, identicalNegatives=True)
            model = models.W2VModel("syriastrikes", sg=sg, window=window, N=750)
            model.overwriteTraindata(clean=train)
            model.build()
            model.computeDocumentVector()

            f = []
            indices = []

            under = 0.7
            upper = 1
            thres = under
            while thres <= upper: 
                indices.append(thres)
                (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
                f.append(classification.f1Measure(rec,prec))

                thres += 0.001

            #ind = np.array(len(indices))
            i += 1

    plt.legend(labels)
    plt.show()

def testdataStemStop():
    labels = ["Halb bereinigt", "Ohne Füllwörter / Ohne Stemming", "Mit Stemming / Mit Füllwörtern"]
    colors = ["r",              "g",               "b"]

    # fullclean/stem/stop
    tests = [(False, False, False), (False, False, True), (False, True, False)]

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))

    i = 0

    model = models.W2VModel("syriastrikes", N=750)
    model.build()
    model.computeDocumentVector()

    for (full,stem,stop) in tests:
        
        testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=full, onlystemming=stem, onlystop=stop,identicalNegatives=True)

        f = []
        indices = []

        under = -0.3
        upper = 0.8
        thres = under
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            f.append(classification.f1Measure(rec,prec))

            thres += 0.001

        #ind = np.array(len(indices))
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(labels)
    plt.show()

def traindataStemStop():
    labels = ["Halb bereinigt", "Ohne Füllwörter / Ohne Stemming", "Mit Stemming / Mit Füllwörtern"]
    colors = ["r",              "g",               "b"]

    # fullclean/stem/stop
    trains = ["cleaned/syriastrikes.halfclean", "cleaned/syriastrikes.d2vclean2", "cleaned/syriastrikes.d2vclean1"]

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))

    i = 0
        
    testdocs = utilities.getLabeledTestdata("syriastrikes", fullclean=True ,identicalNegatives=True)

    for path in trains:
        removePath = "testing/syriastrikes_.word2vec_1_750"
        if(exists(removePath)):
            print("removing " + removePath)
            remove(removePath)

        model = models.W2VModel("syriastrikes", sg=1, window=10, N=750)
        model.overwriteTraindata(path=path)
        model.build()
        model.computeDocumentVector()

        f = []
        indices = []

        under = 0.5
        upper = 1
        thres = under
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            f.append(classification.f1Measure(rec,prec))

            thres += 0.001

        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(labels)
    plt.show()

def negativeSamplingTest(bow):
    hashtags = ["muslimban", "flynn", "syriastrikes"]
    labels = ["#MuslimBan ohne NS", "#MuslimBan mit NS", "#Flynn ohne NS", "#Flynn mit NS", "#SyriaStrikes ohne NS", "#SyriaStrikes mit NS"]
    colors = ["r",                  "m",                 "g",              "y",             "b",                     "c"]

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    if bow:
        sg = 0
        fullclean = False
    else:
        sg = 1
        fullclean = True

    plt.yticks(np.arange(0,1,0.1))
    i = 0

    for hashtag in hashtags:
        removePath = "testing/" + hashtag + "_.word2vec_" + str(sg) + "_750"
        utilities.removeIfExists(removePath)

        for negativeSampling in [False, True]:
            testdocs = utilities.getLabeledTestdata(hashtag, fullclean=fullclean ,identicalNegatives=True)

            model = models.W2VModel(hashtag, useNS=negativeSampling, sg=sg)
            model.build()
            model.computeDocumentVector()
        
            f = []
            indices = []

            under = -0.25
            upper = 1
            thres = under
            while thres <= upper: 
                indices.append(thres)
                (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
                measurement = classification.f1Measure(rec,prec)
                if measurement >= 0.4:
                    f.append(measurement)
                else:
                    f.append(None)

                thres += 0.001

            plt.plot(indices, f, colors[i])
            i += 1

            removePath = "testing/" + hashtag + "_.word2vec_" + str(sg) + "_750"
            utilities.removeIfExists(removePath)

    plt.legend(labels)
    plt.show()
    #plt.savefig("../vorarbeit/experimente/bilder/newPlots/w2v_ns_" + str(sg) + ".png")


def biggerWindowCBOW(const):
    hashtags = ["muslimban", "flynn", "syriastrikes"]
    labels = ["#MuslimBan; w=5", "#MuslimBan; w=10", "#Flynn; w=5", "#Flynn; w=10", "#SyriaStrikes; w=5", "#SyriaStrikes; w=10"]
    colors = ["r",               "m",                "g",           "y",            "b",                  "c"]

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    if const:
        scaling = 0.1
    else:
        scaling = 0

    outputpath = "dataoutput/biggerwindowcbow.txt"
    outputdata = []

    plt.yticks(np.arange(0,1,0.1))
    i = 0

    for hashtag in hashtags:
        removePath = "testing/" + hashtag + "_.word2vec_0_750"
        utilities.removeIfExists(removePath)
        utilities.removeIfExists(outputpath)
        
        for windowSize in [5,10]:
            testdocs = utilities.getLabeledTestdata(hashtag, fullclean=False ,identicalNegatives=True)

            model = models.W2VModel(hashtag, sg=0, useNS=False, window=windowSize, N=750)
            model.build()
            #model.computeDocumentVector()
        
            f = []
            indices = []

            under = 0
            upper = 0.5
            thres = under
            while thres <= upper: 
                indices.append(thres)
                (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
                measurement = classification.f1Measure(rec,prec)
                if measurement >= 0.4:
                    if windowSize == 10:
                        f.append(measurement + scaling)
                    else:
                        f.append(measurement)
                else:
                    f.append(None)

                thres += 0.001

            plt.plot(indices, f, colors[i])
            i += 1

            removePath = "testing/" + hashtag + "_.word2vec_0_750"
            utilities.removeIfExists(removePath)
    
    plt.legend(labels)
    #plt.show()
    plt.savefig("../vorarbeit/experimente/bilder/newPlots/w2v_window_" + str(scaling) + ".png")



def testNewDV():
    colors = ["r", "g"]
    model = models.W2VModel("muslimban")
    model.build()

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))
    i = 0



    for clean in [True, False]:
        model = models.W2VModel("muslimban")
        model.build()

        testdocs = utilities.getLabeledTestdata("muslimban", fullclean=clean ,identicalNegatives=True)
        f = []
        indices = []

        under = 0
        upper = 1
        thres = under
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)

            thres += 0.05

        plt.plot(indices, f, colors[i])
        i += 1
    
    plt.legend("bla, bla")
    plt.show()
