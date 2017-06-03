## README Bachelorarbeit Code 
Zunächst sei angemerkt, dass die Programme nicht auf der DVD ausgeführt werden sollten, da zur Beschleunigung der Rechenzeit die Modelle
gespeichert und geladen werden. Dazu muss sichergegangen sein, dass die Dateien an einer schreibbaren Stelle liegen. Außderdem müssen, die Korpusdateien im *corpora* Ordner, falls diese benutzt werden sollen, zunächst entpackt werden.

Es wurde für diese 
Arbeit `Python 3` verwendet, weshalb auch bei der Ausführung `python3` verwendet werden soll. 

Ausserdem sind alle Pfade relativ zum `code` Ordner gewählt worden. Darum muss, sollte eine Funktion ausgeführt werden, dies aus dem `code` Ordner getan werden 
(z.B. über die Kommandozeile: Näheres zur korrekten Ausführung im Unterpunkt **Experimente**)

### Use case
Der Use case ist ein eigenständiges Programm und kann per Kommandozeilenbefehl `python3 usecase.py` direkt im `code` Ordner ausgeführt werden.

### Modelle
Alle genutzten Modelle finden sich in der `models.py` Datei. Diese haben einen einheitlichen Aufbau. Zunächst wird das Modell mit z.B. 
```python
model = new TFIDFModel(...)
``` 
initialisiert. Dann wird mittels 
```python
model.build()
``` 
entweder ein bereits aufgebautes Modell geladen (falls zuvor eins erstellt wurde) oder es wird neu erstellt.

### Experimente
Die durchgeführten Experimente befinden sich in der `experiments.py` als separate Methoden. Zum Ausführen auf der Kommandozeile:

1. Navigiere in den `code` Ordner
2. Öffne den interaktiven Python Interpreter: `python3`
3. Lade die `experiments.py` Datei: `import experiments`
4. Starte ein beliebiges Experiment, z.B.: `experiments.word2vecFeatureSize(True)` 

### Klassifikation
In der Datei `classification.py` finden sich unter anderem die Methoden, mit der mittels Kosinusähnlichkeit (`cosineSimilarityScores(...)`) oder mittels Lloyd's Algorithmus (`ClusterKMeans(...)`) 
Klassifikation durchgeführt werden kann. Außerdem enthällt diese Datei die Methoden zur Berechnung der *precision*, *recall* und *F1* Werte sowie der Kosinusdistanz zweier Vektoren.

### WebScraper
Unter der Datei `webscraper.py` befinden sich die genutzten Methoden, um die Artikel aus der Artikel-URL zu extrahieren.

### Datenbereinigung
In der Datei `datacleaning.py` befindet sich die Hauptmethode, mit der ein String bereinigt werden kann (`cleanString(...)`). Dies ist Hauptmethode, welche
von den anderen Methoden in der Datei genutzt wurde, um ganze Dokumente zu bereinigen.

### Tweets mitschneiden
Zum Mittschneiden der Tweetes eines bestimmten Hashtags findet sich in der `hashtag_streaming.py` Datei eine geeignete Funktion. Diese benötigt im selben Ordner einen Ordner namens *corpora*, wo die Tweets abgespeichert werden können.

### Sonstiges
Die Datei `utilities.py` enthällt Hilfsmethoden, welche in vielen anderen Methoden genutzt wurden. Unter anderem enthält sie aber auch die Methode `getLabeledTestdata(...)`, welche die Testdaten holt und bereinigt.
