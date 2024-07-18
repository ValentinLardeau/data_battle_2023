# data-battle-2023
Data battle 2023

## Description
Projet pour la Data battle 2023 réalisé par l'équipe Just An Other Team:  
Le but de ce projet est de recevoir des relevés lithographiques de puits au format pdf et de calculer les quantités des matériaux présents dans ces puits.

## Version de python
3.9.16

## Dépendances
- pytesseract
- tensorflow
- matplotlib
- panda
- numpy
- streamlit (pout l'interface)
- open-cv
- keras_ocr
- sklearn
- PIL
- Levenshtein
- AIPowerMeter (pour la consommation)
- pypdfium2

## Préparation
Le pdf doit être placé dans un dossier input à la racine du projet.

## Lancement du script
``` shell
python3 run read_pdf.py -f <chemin_depuis_input/nom_du_fichier.pdf> -f false
```

## Affichage de la consommation
```shell
python3 run enregistrementConso.py
```
Ce script requiert que le dossier NO_Quad_15 soit placé dans input.

## Lancement de l'interface (incomplet)
``` shell
streamlit run interface.py
```
Sélectionnez ensuite le fichier pdf situé directement dans input.
