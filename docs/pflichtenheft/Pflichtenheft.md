# Pflichtenheft

## Rahmenbedingungen
* Projektauftraggeber: Professor Thomas Stütz
* Projektleiter: Erik Mayrhofer
* Projektmitarbeiter: Erik Mayrhofer, Florian Schwarcz
* Ausstattung: Raspberry Pi, PI-Infrarotkamera, RPI Weitwinkel-CAM, Logitech 270 Webcam

## Motivation

Dieses Projekt wird im Rahmen des SYP-Unterrichts durchgeführt und wurde von unserem Professor, Herrn Professor Stütz, in Auftrag gegeben. Wir sollen uns mit Objekt- bzw. Gesichtserkennung auseinandersetzen und somit das Robolab der HTL-Leonding ein Stück sicherer machen.

## Ausgangslage und Ist-Zustand

### Problembereich

In der HTL-Leonding gibt es im Untergeschoss das Robolab. Dort drinnen arbeiten Schüler und Lehrer zum einen an den NAO's (Humanoide Roboter) und zum anderen - unter der Aufsicht von Professor Stütz - an Raspberries und anderen ähnlichen Projekten.

![Plan des Robolabs](./images/Robolab-Plan.jpg "Relevanter Bereich des Robolabs (nicht maßstabsgetreu)")

Da die Tür des Robolabs nicht immer abgesperrt wird und sich zusätzlich fast jeder Schüler Zugriff verschaffen kann, ist die Sicherheit des Robolabs nicht gewährleistet. Wenn Schäden - ob willkürlich oder durch einen Unfall - auftreten, kann zur Zeit nicht nachgewiesen werden, wer dafür verantwortlich ist.

### Glossar

| Begriff | Erklärung
| - | -
| Robolab | Raum, der im Problembereich genau beschrieben wurde
| NAOs | Humanoide Roboter, mit denen unter anderem im Robolab gearbeitet wird
| Raspberry Pi | Minicomputer
| Eintrittsereignis | Betreten des Robolabs
| Winkelagnostizität | Fähigkeit, Gesichter zu erkennen, die nicht zwingend frontal aufgenommen wurden
| Erkennungssicherheit | Wert zur Bestimmung, wie sehr ein erkanntes Gesicht mit einem der Vergleichsbildern übereinstimmt
| Erkennungsgenauigkeit | Erfolgschance, ein Gesicht richtig zuzuordnen

### Robolab Use-Case-Diagramm

![Use-Case-Diagramm des Robolabs](./images/Use-Case-Diagram-Before.jpg "Use-Case-Diagramm des Robolabs ohne Sicherheitssystem")

## Zielsetzung

Die Sicherheit im Robolab soll durch Installation einer Kamera mit Gesichtserkennung erhöht werden. Es sollen Daten über den Aufenthalt von Personen im Raum gesammelt werden. Zu verwenden sind die in der Ausstattung enthaltenen Kameras sowie der Raspberry Pi.

Verwendet wird das System bzw. dessen generiertes Protokoll nur von den Betreibern des Robolabs, die die Aufenthaltsdaten brauchen.

## Sollzustand

Die Software auf dem Raspberry Pi soll Gesichter erkennen und sowohl Daten über die Person, als auch Zeitpunkt der Registrierung in eine Datei speichern. Die Gesichter werden nicht zwingend frontal aufgenommen, demnach muss Winkelagnostizität gegeben sein.

### Funktionale Anforderungen

ID: Anf01: Gesichter erkennen\
ID: Anf02: Gesichter zuordnen\
ID: Anf03: Protokoll erstellen\
ID: Anf04: Protokolle über Fileserver zugänglich machen

### Robolab Soll-Use-Case-Diagramm

![Use-Case-Diagramm des Robolabs](./images/Use-Case-Diagram-After.jpg "Use-Case-Diagramm des Robolabs mit Sicherheitssystem")

### Nicht-Funktionale Anforderungen
Die Erkennungsgenauigkeit soll möglichst hoch sein, als Mindestzielwert wird 90% in Betracht gezogen.
Das System soll nicht überlastet werden, wenn es viele Personen gleichzeitig erkennt und zuordnen muss. Es muss nicht zwingend in Echtzeit die Gesichter zuordnen können.
Auch bei ungünstigen Lichtverhältnissen soll die 90%-Quote eingehalten werden.
Nichterkennungen sollen auch mitprotokolliert werden.

## Mengengerüst
Folgende Stammdaten werden sich ergeben:

Jede Person, die Zutrittsauthorisierung hat, wird eingetragen mit:
* Name
* Klasse
* Gesichtsdaten

Für jedes Eintrittsereignis wird mitprotokolliert:
* Uhrzeit und Datum
* Vermutete Person
* Erkennungssicherheit

## Schnittstellenübersicht
Die Protokolle können über FTP direkt am Raspberry eingesehen werden. Der Raspberry nimmt die nötigen Bilder mit einer Kamera auf. Die Kamera ist möglicherweise infrarot- oder weitwinkelfähig.