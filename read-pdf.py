import argparse
import pypdfium2 as pdfium
import os
import numpy as np

from PIL import Image
import cv2
import pytesseract
import re
import find_legend


def remove_noise(img):
    # Convert the image to a numpy array
    data = np.asarray(img)
    print("Converting to grayscale...")
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    print("Denoising image...")
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray_image, None, 20, 10, 21)
    # Convert the image back to a PIL image
    denoised_img = Image.fromarray(denoised)
    return denoised_img


def find_interesting_pages(pdf):
    interesting_pages = []
    # Loop through the table of contents
    for item in pdf.get_toc():
        # Find the composite log page
        if re.search(r"\bcomposite log\b|\bcompletion log\b|\bcompletion  log\b|\bwell summary\b|\benclosures\b|\bcore description\b", item.title.lower()):
            interesting_pages.append(item.page_index)

    return interesting_pages


def print_toc(pdf):
    for item in pdf.get_toc():

        if item.n_kids == 0:
            state = "*"
        elif item.is_closed:
            state = "-"
        else:
            state = "+"

        if item.page_index is None:
            target = "?"
        else:
            target = item.page_index + 1

        print(
            "    " * item.level +
            "[%s] %s -> %s  # %s %s" % (
                state, item.title, target, item.view_mode,
                [round(c, n_digits) for c in item.view_pos],
            )
        )


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-f", "--file", required=True, help="Path of the file you want to search the composite log")
    argParser.add_argument(
        "-p", "--page", default="None", help="FACULTATIVE : number of the page of the composite log (if you know it) ")
    argParser.add_argument(
        "-a", "--alone", default=True, help="FACULTATIVE : if you want to use only the script (True) or the entire project (anything else)")

    args = argParser.parse_args()

    file_name = args.file
    page = args.page
    alone = args.alone
    n_digits = 3  # number of digits in the page number

    input_path = "./input/"
    output_path = "./output1/"
    # file_name = "NO_Quad_15/15_9-9/15_9-9__WELL__15-09-09_PB-706-0109.pdf"

    max_height = 10000  # max height of the image that will be processed by tesseract

    pdf = pdfium.PdfDocument(input_path + file_name)  # open the document
    n_pages = len(pdf)  # get the number of pages in the document

    if(page == 'None'):
        print_toc(pdf)  # print the table of contents

        interesting_pages = find_interesting_pages(pdf)
        print("Interesting pages : " + str(interesting_pages))
    else:
        page = int(page)
        interesting_pages = [page]
        print("Interesting pages : " + str(interesting_pages))

    page_indices = [i for i in range(n_pages)]  # all pages
    renderer = pdf.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=interesting_pages,
        scale=300/72,  # 300dpi resolution
    )

    image_output_path = output_path + file_name.split(".")[0] + "/images"

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    list_img = []
    for i, image in zip(interesting_pages, renderer):
        # Do not process the page if it has already been processed
        list_img.append(image_output_path + "/%0*d.jpg" % (n_digits, i))
        if not os.path.exists(image_output_path + "/%0*d.jpg" % (n_digits, i)):
            print("Processing page %d..." % i)
            denoised_img = remove_noise(image)

            print("Saving image to %s" %
                  image_output_path + "/%0*d.jpg" % (n_digits, i))
            denoised_img.save(image_output_path + "/%0*d.jpg" % (n_digits, i))
    if alone != True:
        find_legend.lauch(list_img[0], False)
