# Pflichtenheft

## Rahmenbedingungen:
* Projektleiter: Erik Mayrhofer
* Projektmitarbeiter: Erik Mayrhofer, Florian Schwarcz
* Ausstattung: Raspberry Pi, IR-Kamera, Fischaugenkamera, LG Webcam

TODO NUMMERN
## Motivation

Diese Projekt wird im Rahmen von SYP durcheführt und wurde von unserem Professor, Herrn Stütz, in Auftrag gegeben. Wir sollen uns mit Objekt- bzw. Gesichtserkennung auseinandersetzen und somit das Robolab der HTL-Leonding ein Stück sicherer machen.

## Ausgangslage und Ist-Zustand

### Problembereich

In der HTL-Leonding gibt es im Untergeschoss das Robolab. Dort drinnen arbeiten Schüler und Lehrer zum einen an den NAO's (Humanoide Roboter) und zum anderen - under der aufsicht von Professor Stütz - an Raspberries und anderen ähnlichen Projekten.

![Plan des Robolabs](./images/Robolab-Plan.jpg "Relevanter Bereich des Robolabs (nicht maßstabsgetreu)")

Da die Tür des Robolabs nicht immer abgesperrt wird und sich zusätzlich fast jeder Schüler zugriff verschaffen kann ist die Sicherheit der Roboloab nicht gewährleistet. Wenn Schäden - ob willkürlich oder durch einen Unfall - auftreten kann zur Zeit nicht nachgewiesen werden, wer dafür verwantwortlich gemacht werden kann.

### Glossar

Nao / Stütz
Bereiche

### Abläufe

Use-Case vom RoboLab?

## Zielsetzung

Die Sicherheit im Robolab soll durch Installation einer Kamera mit Gesichtserkennung erhöht werden. 90% aller Gesichter sollten richtig erkannt und identifiziert werden, wodurch Daten über den Aufenthalt von Personen im Raum gesammelt werden können. Zu verwenden sind die in der Ausstattung enthaltenen Kameras sowie der Raspberry Pi.
Das System soll auch dazu fähig sein, wenn es am Straßenrand installiert wird, vorbeifahrende Fahrzeuge zu erkennen und den Kategorien PKW und LKW zuzuordnen.

TODO Zielgruppe des Interfaces

## Sollzustand

Die Software auf dem Raspberry Pi soll Gesichter erkennen und sowohl Daten über die Person, als auch Zeitpunkt der Registrierung in eine Textdatei speichern. Die Gesichter sollen nicht zwingend frontal aufgenommen werden müssen, demnach muss Perspektivenunabhängigkeit gegeben sein.
Im Abschnitt der Fahrzeuge soll nur erkannt werden, ob das vorbeifahrende Fahrzeug ein PKW oder LKW ist. Andere Eigenschaften wie Modell, Farbe, Größe u. A. sollen *nicht* beachtet werden.

TODO AUFTEILEN

### Funktionale Anforderungen
Arbeitsaufteilung
#### Use Case

TODO SOll use case diagramm

##### Beschreibung der Use-Cases
### Nicht-Funktionale Anforderungen
Erkennungsgenauigkeit
Performance
Lichtverhältnisse
Fehlerbehandlung

## Mengengerüst
Folgende Stammdaten werden sich ergeben:

Jede Person die Zutrittsauthorisierung hat, wird eingetragen mit:
* Name
* Klasse
* Gesichtsdaten

Für jedes Eintrittsereignis wird mitprotokolliert:
* Uhrzeit und Datum
* Vermutete Person
* Erkennungsgenauigkeit

## Risikoakzeptanz ???

## Schnittstellenübersicht
Die Protokolle können über FTP direkt am Raspberry eingesehen werden. Der Raspberry nimmt die nötigen Bilder mit einer Kamera auf. Die Kamera ist möglicherweise Infrarot oder Weitwinkelfähig.