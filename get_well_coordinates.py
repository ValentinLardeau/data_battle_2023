import cv2
import pytesseract
import re


def reduce_image(img_original, path_name, i=0.0):
    height, width = img_original.shape[:2]
    if (i == 0.0):
        cropped = img_original[0:int(height * 0.05), 0:width]
    else:
        cropped = img_original[int(
            height * i):int(height * (i + 0.05)), 0:width]

    cv2.imwrite(path_name, cropped)
    return cropped


def getCoordinates(img_name):
    img = cv2.imread(img_name)
    hauteur_actuelle, largeur_actuelle, _ = img.shape
    image_recadree = reduce_image(img, "image_recadree.jpg")

    # Conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(image_recadree, cv2.COLOR_BGR2GRAY)

    # Detection de texte
    text = pytesseract.image_to_string(gray)

    # Recherche du mot clé "COORDINATES ou COORDINATE S"
    match = re.search(r'COORDINATE\s?S', text)
    # Recherche des coordonnees à l'aide de l'expression reguliere
    match_x = re.search(r'\d?\d?\d?..........\d.\s?[N]', text)
    match_y = re.search(r'\d?\d?\d?...........\d.\s?[E]', text)
    if match:
        coordinates_text = match.group(0)
        print(coordinates_text)

        if match_x and match_y:
            x_str, y_str = match_x.group(0), match_y.group(0)
            print("Les coordonnées sont :", x_str," et ",y_str)
            return coordinates_text, x_str, y_str
        else:
            print("Les coordonnées non  trouvées dans l'image")
            return coordinates_text
    else:
        print("Le mot clé 'COORDINATES' non trouvé dans l'image")

#result 15_8-1 : Les coordonnées sont : 58°21'48.10"N  et  01° 34'22.31"E
#result 15_3-1 :Les coordonnées sont : 58°50’ 57.004’N  et  O1°.43/13240"E
#result 15_3-1 :Les coordonnées sont : 58° 14' 36,59" N  et  Ol? 52'45,677 E
