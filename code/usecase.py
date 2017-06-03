from models import *
from utilities import getLabeledTestdata
from classification import *
from random import shuffle 
def main():
    HASHTAG = "muslimban"
    print("#"*80)
    print("# Verwendeter Hashtag: " + HASHTAG)
    print("#"*80)
    print("# Verwendetes Modell: TF-IDF")

    # tfidf model laden
    model = TFIDFModel(HASHTAG, verbose=False)
    model.build()

    # passende testdaten laden
    testdata = getLabeledTestdata(HASHTAG, verbose=False, identicalNegatives=True)

    # klassifizieren
    (f1Score, positives, negatives) = ClusterKMeans(model, testdata)

    # wie gut hat der Klassifikator abgeschnitten
    print("#"*80)
    print("# F1 Wert des Klassifikators: " + str("%.3f" % f1Score))

    # zeige einige der positiv klassifizierten Artikel (bzw. deren Überschrift) an
    print("#"*80)
    print("# Folgende Artikel wurden unter anderem zur Recherche empfohlen: ")
    print("# ")
    shuffle(positives)
    for article in positives[:10]:
        if len(article) >= 80:
            print("# " + article[:75] + "...")
        else:
            print("# " + article)
    print("#"*80)
main()