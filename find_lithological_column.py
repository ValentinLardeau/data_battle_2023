import argparse
import keras_ocr
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import cv2
import numpy as np
from PIL import Image
from Levenshtein import distance as levenshtein_distance
import Read_colonne


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Functon for extracting the box
def box_extraction(img, debug=False):
    """
    box_extraction search for all the vertical lines and creates the associate files with all the columns
    :img_for_box_extraction_path img__path: path of the image you want to crop
    """

    print("Searching lines..")

    height, _width = img.shape[:2]

    # Thresholding the image
    (_thresh, img_bin) = cv2.threshold(img, 128, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image
    img_bin = 255-img_bin

    print("Applying Morphological Operations..")

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=2)

    # Morphological operation to detect horizontal lines from an image / the iteration are set to 20 so that there is no horizontal lines
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=20)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=20)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (_thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours for image, which will detect all the boxes (here just lines as we don't have horizontal lines)
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by left to right.
    (contours, _boundingBoxes) = sort_contours(contours)

    line_img = np.zeros_like(img)

    lines_to_keep = []
    for c in contours:

        # gathers all the coordinates of the contours of the lines and sorted them by the height.
        l = []
        for elt in c:
            l.append((elt[0][0], elt[0][1]))
        l.sort(key=lambda a: a[1])

        # saves all the coordinates
        x = [x1 for x1, _ in l]
        y = [y1 for _, y1 in l]

        # if there are more than 10 differents coordinates, then draw the line from the beginning to the end of the doc.
        if(len(x) > 10) and (abs(x[0]-x[len(y)-1]) < 200):
            cv2.line(line_img, (x[0], y[0]),
                     (x[len(y)-1], y[len(y)-1]), 255, 5)
            cv2.line(line_img, (x[0], 0), (x[0], y[0]), 255, 5)
            cv2.line(line_img, (x[len(y)-1], y[len(y)-1]),
                     (x[len(y)-1], height), 255, 5)

            # save the line to keep, for later.
            lines_to_keep.append(((x[0], y[0]), (x[len(y)-1], y[len(y)-1])))

    # For Debugging
    # Enable this line to see all contours.
    if debug:
        print("Storing binary image to Images/Image_bin.jpg..")
        cv2.imwrite("Images/Image_bin.jpg", img_bin)
        cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)
        cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

        print("Binary image which only contains lines: Images/img_final_bin.jpg")
        cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
        cv2.imwrite("./Temp/test.jpg", line_img)

        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imwrite("./Temp/img_contour.jpg", img)
    return lines_to_keep


def reduce_img(img_original, path_name, ocr=None, rotation=None, i=0.0):
    """
    cherche_colonne crops the image by 7.5% of their maximum size.
    :param img__path: path of the image you want to crop
    :param name: path of the output cropped image (example : 'cropped.jpg')
    """
    if(ocr == None):
        print("Reducing image..")
        height, width = img_original.shape[:2]
        if(i == 0.0):
            cropped = img_original[0:int(height*0.05), 0:width]
        else:
            cropped = img_original[int(
                height*i):int(height*(i+0.05)), 0:width]

        cv2.imwrite(path_name, cropped)
        print("Done.")
        return cropped
    else:
        print("Reducing image..")
        _text, box = ocr[0]
        x1, x2, x3, x4 = box[0], box[1], box[2], box[3]
        centre = [(x1[0]+x2[0]+x3[0]+x4[0])/4,
                  (x1[1]+x2[1]+x3[1]+x4[1])/4]
        if(rotation == "ccw"):
            centre = [img_dim[0] - centre[1], centre[0]]

        height, width = img_original.shape[:2]
        print(centre)
        centre[0] = centre[0] + (i-0.05)*height

        lower_bound = int(centre[0]+2000)
        cropped = img_original[int(centre[0]):lower_bound, 0:width]
        cv2.imwrite(path_name, cropped)
        print("Done.")
        return cropped


def enhance(name, name_output):
    """
    enhance enhances the image and saves three pictures, the enhanced image and the clock and anticlockwise enhanced images
    WARNING: When the quality is already very good, it make the quality "too good" and more difficult for the OCR to work.
    I recommend trying with the non enhanced pictures and then the enhanced ones if the text was not found.
    :param name: path of the image you want to enhance
    :param name_output: only the name of the image you want to enhance (example : 'cropped')
    """
    im = Image.open(name)

    im = ImageEnhance.Sharpness(im)
    im = im.enhance(1.3)

    im = ImageEnhance.Contrast(im)
    im = im.enhance(5)

    im = ImageEnhance.Brightness(im)
    im.enhance(5).save(name_output + "_enhanced.jpg")
    return name_output + "_enhanced"


def rotate(name, direction):
    img = cv2.imread(name + ".jpg", 0)

    # rotating the image 90 or -90°
    if(direction == "cw"):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(direction == "ccw"):
        img = cv2.rotate(
            img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(name + "_" + direction + ".jpg", img)
    return(name + "_" + direction + ".jpg", img.shape)


def OCR(picture, debug=False):
    """
    OCR apply keras_ocr to detect the text inside the picture. 

    :param picture: string 
    """

    pipeline = keras_ocr.pipeline.Pipeline()

    images = [
        keras_ocr.tools.read(picture)
    ]

    prediction_groups = pipeline.recognize(images)

    # Debug
    if(debug):
        _fig, axs = plt.subplots(nrows=1, figsize=(10, 20))
        keras_ocr.tools.drawAnnotations(image=images[0],
                                        predictions=prediction_groups[0],
                                        ax=axs)
        plt.show()
    return prediction_groups[0]


def find_closest_line(centre, lines):
    '''
    find_closest_line finds the 2 closest lines to the centre of the text
    to do that, it will sort the lines by their x coordinate and then take the two closest to the centre
    :param centre: the centre of the text
    :param lines: all the vertical lines (list of 4 float) 
    '''

    closest_lines = {}
    for line in lines:
        moyenne_x = (line[0][0]+line[1][0])/2
        closest_lines[line] = abs(moyenne_x-centre[0])

    closest_lines = sorted(closest_lines, key=closest_lines.get)

    above = (0, 0)
    below = (0, 0)
    for line in closest_lines:
        moyenne_x = (line[0][0]+line[1][0])/2
        if moyenne_x < centre[0] and above == (0, 0):
            above = line
        elif moyenne_x > centre[0] and below == (0, 0):
            below = line

    return [above, below]


def searchLithoColumn(predicted_image, lines, img_dim, rotation):
    '''
    searchLithoColumn search for the column of lithology (not doing it right now, need to search through lines)
    :param predicted_image: the different text predicted with their associated coordinates
    :param lines: all the vertical lines (list of 4 float) 
    return the tuple of the text found in predicted_image and the coordinates of the closest line of this text
    '''

    liste_debut_column = ["lithology", "file", "litho", "section", "lithological", "column", "logical",
                          "depth", "depths",  "metre"]
    texte_ligne = []

    for text, box in predicted_image:
        for word in liste_debut_column:
            if text in liste_debut_column or (levenshtein(text, word) < 2 and (len(text) >= len(word)-1)):
                x1, x2, x3, x4 = box[0], box[1], box[2], box[3]
                centre = ((x1[0]+x2[0]+x3[0]+x4[0])/4,
                          (x1[1]+x2[1]+x3[1]+x4[1])/4)
                if(rotation != "None"):
                    centre_rotated = [img_dim[0] - centre[1], 0]
                    closest_line = find_closest_line(centre_rotated, lines)
                    texte_ligne.append(
                        (text, closest_line[0], closest_line[1], box))
                else:
                    closest_line = find_closest_line(centre, lines)
                    texte_ligne.append(
                        (text, closest_line[0], closest_line[1], box))

    return texte_ligne


def levenshtein(mot1: str, mot2: str) -> int:
    """
    Do the Levenshtein distance between the two words.
    The Levenshtein distance is the minimal number of addition, deletion or
    substitutions of characters to do to go from one word to an other.
    :param mot1: First word to compare.
    :param mot2 : Second word to compare.
    return Levenshtein distance between the two given words.
    """

    return levenshtein_distance(mot1, mot2)


def returnColumn(img_path, output_path, column, i, begin=0):
    '''
    returnColumn save the column  if text could be in liste_beginning_column using regular expression
    :param text:
    :param liste_beginning_column: list of word that are the beginning of the column we search. 
    return True if text could be part of the list.
    '''

    msc = column[0]
    _text, coord1, coord2, box = msc
    coord1_min, coord1_max = coord1
    coord2_min, coord2_max = coord2
    x1_min, y1_min = coord1_min
    x1_max, y1_max = coord1_max
    x2_min, y2_min = coord2_min
    x2_max, y2_max = coord2_max
    xmin = min(x1_min, x1_max, x2_min, x2_max)
    xmax = max(x1_min, x1_max, x2_min, x2_max)

    ymax = box[0][1]
    for j in range(len(box)):
        if(box[j][1] > ymax):
            ymax = box[j][1]

    img = cv2.imread(img_path, 0)
    height, _width = img.shape[:2]
    print(begin)
    print(ymax)
    print(int(height*(i-0.05)))

    # we're adding 100 in each side because the column is not always a straight line
    cropped = img[(int(begin) + int(ymax) + int(height*(i-0.05)))
                   :height, xmin-100:xmax+100]
    print(output_path.split('/')[-1])
    cv2.imwrite("res_" + output_path.split('/')[-1], cropped)


def testOcr(predicted_image):
    '''
    testOcr search if there is a word in predicted_image that fit with one word in list_deb_column
    :param predicted_image:
    return True if there is a word that is similar to one of the word of the list, otherwise False

    '''
    list_deb_column = ["lithology",  "file", "litho", "section", "lithological", "column", "logical",
                       "depth", "depths",  "metre", ]

    for text, _box in predicted_image:
        for word in list_deb_column:
            if text in list_deb_column or (levenshtein(text, word) < 2 and (len(text) >= len(word)-1)):
                return True
    return False


def reduceFrom(h, img_original):
    print("reduceFrom")
    height, width = img_original.shape[:2]
    cropped = img_original[int(h):int(height), 0:width]
    cv2.imwrite("cropped1.jpg", cropped)
    print("Done.")
    return cropped


def launch(image_path, rotation="None", begin=0):
    '''
    Launch the script entirely like it was call using the terminal
    :param image_path: path of the image one would like to search the column
    :param rotation: the needed rotation of the image for finding the name of the column
    :param begin: the number of pixel where one would like to start searching for the column
    '''
    img_original = cv2.imread(image_path, 0)
    img_original = reduceFrom(begin, img_original)

    img = reduce_img(img_original, "cropped1.jpg")
    name_output_enhanced = enhance("cropped1.jpg", "cropped1")
    name, img_dim = rotate(name_output_enhanced, rotation)
    ocr = OCR(name)
    i = 0.05
    res = testOcr(ocr)
    while(not res):
        img = reduce_img(img_original, "cropped1.jpg", None, None, i)
        name_output_enhanced = enhance("cropped1.jpg", "cropped1")
        name, img_dim = rotate(name_output_enhanced, rotation)
        ocr = OCR(name)
        res = testOcr(ocr)
        i += 0.05

    img = reduce_img(img_original, "cropped1.jpg", ocr, rotation, i)

    lines = box_extraction(img)

    result = searchLithoColumn(ocr, lines, img_dim, rotation)
    print(result)

    returnColumn(image_path, image_path, result, i, begin)
    Read_colonne.launch()


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i", "--img", required=True, help="Path of the image you want to search the column in")
    argParser.add_argument(
        "-r", "--rotate", default="None", help="FACULTATIVE : Possibles values: ccw, cw, None. if the image needs to be rotated counter clock wise, clock wise or not. Default is no rotation.")

    argParser.add_argument(
        "-b", "--begin", default=0, help="FACULTATIVE : Beginning of the height that we search for the column. Default is 0.")

    args = argParser.parse_args()

    image_path = args.img
    img_original = cv2.imread(image_path, 0)
    img_original = reduceFrom(args.begin, img_original)

    rotation = args.rotate

    img = reduce_img(img_original, "cropped1.jpg")
    name_output_enhanced = enhance("cropped1.jpg", "cropped1")
    name, img_dim = rotate(name_output_enhanced, rotation)
    ocr = OCR(name)
    i = 0.05
    res = testOcr(ocr)
    while(not res):
        img = reduce_img(img_original, "cropped1.jpg", None, None, i)
        name_output_enhanced = enhance("cropped1.jpg", "cropped1")
        name, img_dim = rotate(name_output_enhanced, rotation)
        ocr = OCR(name)
        res = testOcr(ocr)
        i += 0.05

    img = reduce_img(img_original, "cropped1.jpg", ocr, rotation, i)

    lines = box_extraction(img)

    result = searchLithoColumn(ocr, lines, img_dim, rotation)
    # print(result)

    returnColumn(args.img, args.img, result, i, int(args.begin))
    Read_colonne.launch()
