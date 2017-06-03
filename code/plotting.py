import models
import matplotlib.pyplot as plt
import classification
from utilities import *
from sklearn import decomposition
from sklearn.cluster import KMeans 


def plotOneDimension():
    model = models.TFIDFModel("syriastrikes")
    model.build()

    testdocs = getLabeledTestdata("syriastrikes", identicalNegatives=True)

    vectors = []
    sim_min = 1
    sim_max = 0

    full_articles = []

    n_positives = 0
    n_negatives = 0

    for (label, cleaned, indicator) in testdocs:
        similarity = classification.cosine_similarity(model.documentVector, model.infer(cleaned))
        if similarity > sim_max:
            sim_max = similarity
        if similarity < sim_min:
            sim_min = similarity
        #vectors.append([0, similarity])
        vectors.append([similarity])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=1, algorithm="full", init=np.array([[sim_min], [sim_max]])).fit(vectors)

    fig, ax = plt.subplots()
    #plt.figure()
    ax.yaxis.set_visible(False)
    #plt.hlines(1,sim_min - 0.01 ,sim_max + 0.01)
    plt.xlabel("Kosinusabstand zum Korpusvektor")
    plt.scatter(x=sim_min, y=1, color="b")
    plt.scatter(x=sim_max, y=1, color="m")
    #plt.eventplot([sim_min], color="b", linelength=0.5)
    #plt.eventplot([sim_max], color="m", linelength=0.5)
    i = 0
    clusters = kmeans.labels_
    for point in clusters:
        if vectors[i][0] != sim_min and vectors[i][0] != sim_max:
            if clusters[i] == 1:
                color = "g"
            else:
                color = "r"
            #plt.eventplot([vectors[i][0]], orientation="horizontal", color=color, linelength=0.5)
            plt.scatter(x=vectors[i][0], y=1, color=color)
        i += 1
    
    #plt.axis("off")
    plt.show()

    

def visualizePCA():
    model = models.TFIDFModel("flynn", verbose=False)
    model.build()
    model.computeDocumentVector()

    testdocs = getLabeledTestdata("flynn", verbose=False)

    vectors = []
    vectors.append(model.documentVector)
    for (label, clean, ind) in testdocs:
        vectors.append(model.infer(clean))

    vectors = np.array(vectors)
    pca = decomposition.PCA(n_components=2)
    pca.fit(vectors)

    vectors = pca.transform(vectors)

    x = vectors[:,0]
    y = vectors[:,1]

    fig, ax = plt.subplots()
    plt.scatter(x=x[0], y=y[0], color="b")
    plt.scatter(x=x[1:19], y=y[1:19], color="g")
    plt.scatter(x=x[20:], y=y[20:], color="r")
    plt.legend(["corpus vector","positives","negatives"])
    plt.show()

def plotOneW2V(exp, name):
    model = models.W2VModel(name, sg=1, window=10, N=750)
    model.build()
    model.computeDocumentVector()

    testdocs = getLabeledTestdata(name)
    f = []

    indices = []
    thres = 0.7
    while thres <= 1:
        (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
        indices.append(thres)
        f.append(classification.f1Measure(rec, prec))

        thres += 0.001

    fig, ax = plt.subplots()
    plt.xticks(np.arange(0,1,0.05))
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0,1,0.1))
    plt.plot(indices, f, "r")
    #plt.savefig("../vorarbeit/experimente/bilder/" + exp + ".png")
    plt.show()



def plotTFIDF(exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]

    i = 0
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0.4,1,0.05))


    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=True ,identicalNegatives=True)

        model = models.TFIDFModel(hashtag)
        model.build()
    
        f = []
        indices = []

        under = 0
        upper = 1
        thres = under
        LOG("Start measuring")
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)

            thres += 0.001
        LOG("Start plotting...")
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(legend)
    #plt.show()
    plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "tfidf" + ".png")

def plotLDA_KMEANS(exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]

    i = 0
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    #plt.xlabel('Hashtags')
    plt.yticks(np.arange(0,1,0.05))

    f = []


    width = 0.4

    ind = np.arange(len(hashtags))

    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=True ,identicalNegatives=True)

        model = models.LDAModel(hashtag)
        model.build()

        (f1score, positives, negatives) = classification.ClusterKMeans(model, testdocs)

        #plt.barh(ind[i], f1score, width, color=colors[i])
        f.append(f1score)
        i += 1

        #bars = ax.bar(np.arrange(len(hashtags)), f1score, width, color=colors[i], yerr=men_std)
    

    rects = ax.bar(ind, f, width, color=colors)
    ax.set_xticks(ind)
    ax.set_xticklabels(legend, minor=False)
    
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                "{0:.3f}".format(f[i]),
                ha='center', va='bottom')
        i += 1


    #plt.legend(legend)
    #plt.show()
    plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "lda" + ".png")

def plotTFIDF_KMEANS(exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]

    i = 0
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    #plt.xlabel('Hashtags')
    plt.yticks(np.arange(0,1,0.05))

    f = []


    width = 0.4

    ind = np.arange(len(hashtags))

    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=True ,identicalNegatives=True)

        model = models.TFIDFModel(hashtag)
        model.build()

        (f1score, positives, negatives) = classification.ClusterKMeans(model, testdocs)

        #plt.barh(ind[i], f1score, width, color=colors[i])
        f.append(f1score)
        i += 1

        #bars = ax.bar(np.arrange(len(hashtags)), f1score, width, color=colors[i], yerr=men_std)
    

    rects = ax.bar(ind, f, width, color=colors)
    ax.set_xticks(ind)
    ax.set_xticklabels(legend, minor=False)
    
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                "{0:.3f}".format(f[i]),
                ha='center', va='bottom')
        i += 1


    #plt.legend(legend)
    plt.show()
    #plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "tfidf" + ".png")

def plotW2V_KMEANS(bow, exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]

    i = 0
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    #plt.xlabel('Hashtags')
    plt.yticks(np.arange(0,1,0.05))

    f = []

    if bow:
        sg = 0
        clean = False
    else:
        sg = 1
        clean = True


    width = 0.4

    ind = np.arange(len(hashtags))

    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=clean ,identicalNegatives=identicalNegatives)

        model = models.W2VModel(hashtag, sg=sg)
        model.build()

        (f1score, positives, negatives) = classification.ClusterKMeans(model, testdocs)

        #plt.barh(ind[i], f1score, width, color=colors[i])
        f.append(f1score)
        i += 1

        #bars = ax.bar(np.arrange(len(hashtags)), f1score, width, color=colors[i], yerr=men_std)
    

    rects = ax.bar(ind, f, width, color=colors)
    ax.set_xticks(ind)
    ax.set_xticklabels(legend, minor=False)
    
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                "{0:.3f}".format(f[i]),
                ha='center', va='bottom')
        i += 1


    #plt.legend(legend)
    plt.show()
    #plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "w2v" + str(bow) + ".png")


def plotLDA(exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]

    i = 0
    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0.4,1,0.05))


    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=True ,identicalNegatives=True)

        model = models.LDAModel(hashtag)
        model.build()
    
        f = []
        indices = []

        under = 0
        upper = 1
        thres = under
        LOG("Start measuring")
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)

            thres += 0.001
        LOG("Start plotting...")
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(legend)
    #plt.show()
    plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "lda" + ".png")

def plotW2V(bow, exp="NEW", identicalNegatives=True):
    hashtags = ["muslimban", "flynn", "jointaddress", "killthebill", "syriastrikes"]
    colors   = ["r"        , "g"    , "b"           , "m"          , "y"]
    legend = ["#MuslimBan",  "#Flynn","#JointAddress","#KillTheBill","#SyriaStrikes"]
    if bow:
        sg = 0
        fullclean = False
    else:
        sg = 1
        fullclean = True

    i = 0

    fig, ax = plt.subplots()
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$\tau$')

    plt.yticks(np.arange(0.4,1,0.05))

    for hashtag in hashtags:
        testdocs = getLabeledTestdata(hashtag, fullclean=fullclean ,identicalNegatives=True)

        model = models.W2VModel(hashtag, sg=sg, )
        model.build()
    
        f = []
        indices = []

        under = 0
        upper = 1
        thres = under
        LOG("Start measuring")
        while thres <= upper: 
            indices.append(thres)
            (scores, (rec, prec)) = classification.cosineSimilarityScores(testdocs, model, thres)
            measurement = classification.f1Measure(rec,prec)
            if measurement >= 0.4:
                f.append(measurement)
            else:
                f.append(None)

            thres += 0.001
        LOG("Start plotting...")
        plt.plot(indices, f, colors[i])
        i += 1

    plt.legend(legend)
    plt.show()
    #plt.savefig("../vorarbeit/experimente/bilder/final/" + exp + "w2v" + str(bow) + ".png")

def plotWithRandomTestdata(name, exp="EX"):
    (indices, tfidfF1, w2vcbowF1, w2vskipF1) = classification.getFitnessWithRandomTestdata(name, iterations=50)

    fig, ax = plt.subplots()
    #plt.xticks(np.arange(0,1,0.1))
    plt.ylabel(r'$F_1$')
    plt.xlabel(r'$x$')

    plt.yticks(np.arange(0.5,1,0.05))
    plt.plot(indices, tfidfF1, "r")
    plt.plot(indices, list(map(lambda x: x + 0.1, w2vcbowF1)), "b")
    plt.plot(indices, list(map(lambda x: x + 0.1, w2vskipF1)), "g")
    plt.legend(["TF-IDF", "word2vec CBOW", "word2vec SKIP"])
    plt.savefig("../vorarbeit/experimente/bilder/" + exp + "fitness.png")
