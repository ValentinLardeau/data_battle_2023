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
    
def extract_name(path):
	img = cv2.imread(path)

	hauteur_actuelle, largeur_actuelle, _ = img.shape
	image_recadree = reduce_image(img, "image_recadree.jpg")

	gray = cv2.cvtColor(image_recadree, cv2.COLOR_BGR2GRAY)

	text = pytesseract.image_to_string(gray)

	# Recherche du nom puits
	matches = re.findall(r'.+\s?\/\s?.+\s?-\s?.', text)

	if len(matches) > 0:
	    first_match = matches[0]
	    return first_match


