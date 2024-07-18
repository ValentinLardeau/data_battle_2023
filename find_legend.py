from Levenshtein import distance as levenshtein_distance
import argparse
import keras_ocr
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import cv2
import numpy as np
from PIL import Image
import os
import find_lithological_column

rock_dictionnary = ['sandstone', 'sand', 'siltstone', 'silt', 'shale', 'clay', 'limestone', 'dolomite', 'marl', 'marlstone', 'chalk', 'coal', 'basalt', 'andesite', 'dacite', 'rhyolite',
                    'granite', 'gneiss', 'schist', 'quartzite', 'slate', 'conglomerate', 'breccia', 'tuff', 'volcanic', 'igneous', 'sedimentary', 'metamorphic', 'igneous',
                    'conglomerate', 'gypsum', 'salt', 'silty', 'calcarious', 'claystone', 'rock', 'anhydrite', 'lignite', 'limey', 'dolomitic', 'halite', 'arenaceous', 'mica', 'argillaceous',
                    'kaolinite', 'calcareous', 'dolomitic', 'shert', 'tuffaceous', 'nodules', 'carbonaceous', 'macrofossils', 'microfossils', 'bituminous', 'foraminifera', 'plant', 'pyritic',
                    'sandy', 'silty', 'glauconite', 'pyrite', 'chert', 'fractures', 'stylolites', 'fossils', 'wood', "forams", "siderite", "shell", "burrows", "burows"
                    ]


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
def box_extraction(img_for_box_extraction_path, img_original, zero, output_path="./cropped/", debug=False):
    """
    box_extraction search for all the vertical lines and creates the associate files with all the columns
    :img_for_box_extraction_path img__path: path of the image you want to crop
    """

    print("Reading image..")

    # make the contrast better to obtain better result
    im = Image.open(img_for_box_extraction_path)

    l = img_for_box_extraction_path.split(".")
    name = l[0] + "_enhanced." + l[1]
    im = ImageEnhance.Contrast(im)
    im = im.enhance(20)
    im.save(name)

    img = cv2.imread(name, 0)

    height, width = img.shape[:2]

    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image
    img_bin = 255-img_bin

    print("Applying Morphological Operations..")

    kernel_length = np.array(img).shape[1]//80

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

    # Morphological operation to detect horizontal lines from an image / the iteration are set to 20 so that there is no horizontal lines
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours for image, which will detect all the boxes (here just lines as we don't have horizontal lines)
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by left to right.
    (contours, boundingBoxes) = sort_contours(contours)

    print("Output stored in cropped directory!")

    box_to_keep = []
    idx = 0
    for c in contours:

        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        #  save boxes in "cropped/" folder.
        if (w > 80 and h > 20) and (w < 2000 and h < 200) and w > h and w < 3*h:
            idx += 1
            new_img = img_original[zero+y:zero+y+h, x:x+w]
            box_to_keep.append((x+w/2, y+h/2, idx, w))
            cv2.imwrite(output_path+str(idx) + '.png', new_img)

    # For Debugging
    if(debug):
        print("Storing binary image to Images/Image_bin.jpg..")
        cv2.imwrite("Images/Image_bin.jpg", img_bin)
        cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)
        cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

        print("Binary image which only contains boxes: Images/img_final_bin.jpg")
        cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imwrite("./Temp/img_contour.jpg", img)

    return box_to_keep


def reduce_page(img_original, path_name, i=0.0):
    """
    reduce_page crops the image by 10% of they maximum size.
    :param img__path: path of the image you want to crop
    :param name: path of the output cropped image (example : 'cropped.jpg')
    """

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


def enhance(path_input, path_output, box=False):
    """
    enhance enhances the image and saves three pictures, the enhanced image and the clock and anticlockwise enhanced images
    :param path_input: path of the image you want to enhance
    :param path_output: only the name of the image you want to enhance (example : 'cropped')
    """
    im = Image.open(path_input)
    if(box):
        im = ImageEnhance.Sharpness(im)
        im = im.enhance(0.8)

        im = ImageEnhance.Contrast(im)
        im = im.enhance(40)

        im = ImageEnhance.Brightness(im)
        im = im.enhance(10)
    else:
        im = ImageEnhance.Sharpness(im)
        im = im.enhance(1.3)

        im = ImageEnhance.Contrast(im)
        im = im.enhance(5)

        im = ImageEnhance.Brightness(im)
        im = im.enhance(5)
    im.save(path_output)
    return path_output


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
        fig, axs = plt.subplots(nrows=1, figsize=(10, 20))
        keras_ocr.tools.drawAnnotations(image=images[0],
                                        predictions=prediction_groups[0],
                                        ax=axs)
        plt.show()
    return prediction_groups[0]


def search_legend(predicted_image, boxes):
    '''
    searchLegend search for the text associated to each legend
    :param predicted_image: the different text predicted with their associated coordinates
    :param boxes: all the legends boxes
    '''

    keep_text = []
    for x, y, idx, w in boxes:
        best_x, best_y, distance_min, text_to_keep = 0, 0, np.inf, ""
        for text, box in predicted_image:
            x_box = (box[0][0] + box[0][0] + box[0][0] + box[0][0])/4
            y_box = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
            dist = distance(x, y, x_box, y_box)
            if(x < x_box - w/2):
                if dist < distance_min and len(text) > 1:
                    best_x, best_y, distance_min, text_to_keep = x_box, y_box, dist, text
        bool = True
        for i in range(idx):
            if((i, text_to_keep) in keep_text):
                bool = False

        # This word will be compared to the dictionnary of rocks. If it is not in the dictionnary, but close enough, it will be corrected. If it is too far it will be replaced by an empty string
        corrected_word = word_close_to_dict(text_to_keep, rock_dictionnary, 2)

        if bool and corrected_word != "":
            keep_text.append((idx, corrected_word))
    return keep_text


def distance(x, y, x1, y1):
    return np.sqrt((y1-y)**2 + (x1-x)**2)


def levenshtein(mot1, mot2):
    """
    Returns the Levenshtein distance between the two words.
    The Levenshtein distance is the minimal number of addition, deletion or
    substitutions of characters to do to go from one word to an other.
    Args:
        mot1 (str): First word to compare.
        mot2 (str): Second word to compare.
    Returns:
        int: Levenshtein distance between the two given words.
    """
    return levenshtein_distance(mot1, mot2)


def word_close_to_dict(word, dictionnary, threshold):
    """
    Returns the word in the dictionary that is the closest to the given word.
    Args:
        word (str): Word to compare.
        dictionnary (list[str]): List of words to compare to.
        threshold (int): Threshold of the Levenshtein distance. If the distance is greater than the threshold, the function returns an empty string.
    Returns:
        str: Word in the dictionary that is the closest to the given word.
    """

    distance_min = 100
    word_close = ""
    for word_dict in dictionnary:
        distance = levenshtein(word, word_dict)
        if (distance < distance_min):
            distance_min = distance
            word_close = word_dict
    if distance_min > threshold:
        # print("-Word : " + word + " is not in the dictionnary. The closest word is : " + word_close + " the distance is " + str(distance_min))
        return ""
    else:
        # print("-Word : " + word + " corrected to : " + word_close)
        return word_close


def find_legend_in_prediction(listeMotsLegende, prediction):
    """
    Find the Legend in the document.
    Args:

        listMotsLegende (str): List of words describing a title of legend.
    Returns:
        list[tupple[str, numpy.array]]: List of boxes containing a word of the
        legend.
    """

    for (word, box) in prediction:
        for mot in listeMotsLegende:
            distance = levenshtein(word, mot)
            if (distance <= 1.5):
                return box
    return []


def adapt_image(img_original, dim, box, path_name, i):
    """
    Crop the image to keep only the legend.
    Args:
        img_original: the complete image of the composite log
        dim: the dimensions of the image
        box: the box where the text was found
        path_name: the path where the image will be saved
        i: current index
    Returns:
        the y coordinate at the start of the legend
    """
    height, width = dim
    x1, x2, x3, x4 = box[0], box[1], box[2], box[3]
    centre = ((x1[0]+x2[0]+x3[0]+x4[0])/4,
              (x1[1]+x2[1]+x3[1]+x4[1])/4)
    cropped = img_original[int(
        height*(i-0.05)+centre[1]):int(
        height*(i-0.05)+centre[1]) + 2000, 0:width]

    cv2.imwrite(path_name, cropped)
    return int(height*(i-0.05)+centre[1])


def post_treatment(legend, cropped_path):
    """
    Remove the images which have incorrect text, and crop the images to remove the border and rename them with the name of the rock.
    Args:
        legend: the legend of the composite log
        cropped_path: the path where the output images will be saved
    """
    for path in os.listdir(cropped_path):
        if path.split(".")[0] not in legend[:, 0]:
            # Remove the images which have incorrect text
            os.remove(cropped_path + path)

    for path in os.listdir(cropped_path):
        img = cv2.imread(cropped_path + path, 0)

        # Get the image dimensions
        height, width = img.shape[:2]

        # Set the new starting and ending coordinates of the image
        start_row, start_col = int(10), int(10)
        end_row, end_col = int(height - 10), int(width - 10)

        cropped_img = img[start_row:end_row, start_col:end_col]
        index_legend = 0
        # search the index the image in the array to find out the name of the rock
        for i in range(len(legend)):
            if path.split(".")[0] == legend[i, 0]:
                index_legend = i
                break
        cv2.imwrite(
            str(cropped_path + str(legend[index_legend, 1]) + ".png"), cropped_img)
        print(str(legend[index_legend, 1]))

    for path in os.listdir(cropped_path):
        if path.split(".")[0] in legend[:, 0]:
            # Remove the images which have correct text
            os.remove(cropped_path + path)


def find_legend(img_original, keywords, cropped_image_path, cropped_enhanced_image_path):
    """
    Find the legend in the document.
    Args:
        img_original: the complete image of the composite log
        keywords: the keywords to find the legend
        cropped_image_path: the path where the cropped image will be saved
        cropped_enhanced_image_path: the path where the enhanced cropped image will be saved
    Returns:
        list[tupple[str, numpy.array]]: List of boxes containing a word of the the legend
    """
    reduce_page(img_original, cropped_image_path)
    enhance(cropped_image_path, cropped_enhanced_image_path)

    i = 0.05
    ocr = OCR(cropped_enhanced_image_path)
    res = find_legend_in_prediction(keywords, ocr)

    while len(res) == 0 and i < 1:
        reduce_page(img_original, cropped_image_path, i)
        enhance(cropped_image_path, cropped_enhanced_image_path)
        ocr = OCR(cropped_enhanced_image_path)
        res = find_legend_in_prediction(keywords, ocr)
        i += 0.05

    return res, i


def launch(img, alone=True):

    img_path = img
    keywords = ["symbols", "legend", "ornament"]
    img_original = cv2.imread(img_path, 0)
    img_dim = img_original.shape

    cropped_image_path = "cropped1.jpg"
    cropped_enhanced_image_path = "cropped1_enhanced.jpg"
    output_path = "cropped/"

    # Find the legend in the document
    res, i = find_legend(img_original, keywords,
                         cropped_image_path, cropped_enhanced_image_path)
    # Crop the image to keep only the legend
    zero = adapt_image(img_original, img_dim, res, cropped_image_path, i)
    # Enhance the image
    enhance(cropped_image_path, cropped_enhanced_image_path, True)
    # Extract the boxes of the legend
    boxes = box_extraction(cropped_enhanced_image_path,
                           img_original, zero, output_path)
    # Extract the text of the legend
    ocr = OCR(cropped_enhanced_image_path)
    # Associate the text to the boxes
    legend = np.array(search_legend(ocr, boxes))
    # print the legend
    print(str(legend))
    # Remove the images which have incorrect text, and crop the images to remove the border and rename them with the name of the rock
    post_treatment(legend, output_path)
    if not alone:
        find_lithological_column.lauch(img_path, begin=zero)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--img", required=True, help="Path of the image you want to search the legend of"
    )
    arg_parser.add_argument(
        "-a", "--alone", default=True, help="Path of the image you want to search the legend of"
    )
    args = arg_parser.parse_args()

    img_path = args.img
    keywords = ["symbols", "legend", "ornament"]
    img_original = cv2.imread(img_path, 0)
    img_dim = img_original.shape

    cropped_image_path = "cropped1.jpg"
    cropped_enhanced_image_path = "cropped1_enhanced.jpg"
    output_path = "cropped/"

    # Find the legend in the document
    res, i = find_legend(img_original, keywords,
                         cropped_image_path, cropped_enhanced_image_path)
    # Crop the image to keep only the legend
    zero = adapt_image(img_original, img_dim, res, cropped_image_path, i)
    # Enhance the image
    enhance(cropped_image_path, cropped_enhanced_image_path, True)
    # Extract the boxes of the legend
    boxes = box_extraction(cropped_enhanced_image_path,
                           img_original, zero, output_path)
    # Extract the text of the legend
    ocr = OCR(cropped_enhanced_image_path)
    # Associate the text to the boxes
    legend = np.array(search_legend(ocr, boxes))
    # print the legend
    print(str(legend))
    # Remove the images which have incorrect text, and crop the images to remove the border and rename them with the name of the rock
    post_treatment(legend, output_path)

    if args.alone != True:
        find_lithological_column.lauch(img_path, begin=zero)
