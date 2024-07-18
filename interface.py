import streamlit as st
from find_lithological_column import launch as read_colonne
from extract_well_name import extract_name
from get_well_coordinates import getCoordinates
from find_legend import launch as find_legend
import numpy as np
import os
import matplotlib.pyplot as plt

""" Launch a web application allowing the user to:
- Select a file located in the input directory,
- Lauch the program for this file,
- See the progression of the program includint the name and coordinates of the well,
- Update the pie-chart (unfinished).

This file is not used when launching the script directly.
"""
st.title("Just an other team")

# Section to enter files
st.subheader("Upload file:")
uploaded_file = st.file_uploader(label="Choose a file :")

# Section to display progression
st.subheader("Progression of program:")
displayed_text = "Waiting for a file"
text_progression = st.text(displayed_text)

# Section to display the result
st.subheader("Result:")
# Default data of the diagram
labels = ['A', 'B', 'C', 'D']
values = [25, 35, 20, 10]
# Create the diagram
fig, ax = plt.subplots()
ax.set_title("Diagramme")
ax.pie(values, labels=labels, autopct='%1.1f%%')
# Display the diagram
plot = st.pyplot(fig)

def modif_chart(labels, values):
    """ Change the diagram with the given data.

    Args:
        labels (list[str]): List of labels of the diagram.
        values (list[double]): List of values of the diagram.
    """
    global ax, fig, plot
    plot.empty()
    ax.pie(values, labels = labels, autopct = '1.1f%%')
    plot = st.pyplot(fig)

def modif_chart_image():
    """ Change the diagram to use the pie-chart.jpg image.
    """
    global plot
    plot.empty()
    plot = st.image(Image.open('./output1/pie-chart.jpg'))

def reinit_progress():
    """ Empty the progression indication text.
    """
    global text_progression
    global displayed_text
    displayed_text = ""
    ajouter_progress("")

def ajouter_progress(new_text: str) :
    """ Add text to indicate the progression of the algorithm.

    Args:
        new_text (str): Text to add.
    """
    global text_progression
    global displayed_text
    displayed_text += "\n" + new_text
    text_progression.text(displayed_text)

def modifier_titre(mot: str):
    """ Modify the title of the diagram.

    Args:
        mot (str): New title of the diagram.
    """
    global ax
    ax.set_title(mot)

def executeProgram(file: str):
    """ Execute the program for the given file.

    Args:
        file (str): Name of a pdf file in the input directory.
    """
    reinit_progress()
    ajouter_progress(f"Reading {file}...")
    os.system("python3 read-pdf.py -f \"" + file + "\"")
    ajouter_progress(f"File read.")
    file = file.split(".")[0]
    chemin_image = f"./output1/{file}/images/" + os.listdir(f"./output1/{file}/images")[0]
    ajouter_progress(f"Searching info about the well...")
    nom_puit = extract_name(chemin_image)
    modifier_titre(nom_puit)
    ajouter_progress(f"The well is named {nom_puit}.")
    coordonnees_puit = getCoordinates(chemin_image)
    try:
        ajouter_progress(f"The coordinates are {coordonnees_puit[1]} and {coordonnees_puit[2]}")
    except :
        ajouter_progress(f"The coordinates are not found.")
    ajouter_progress("Reading the legend...")
    try:
        find_legend(chemin_image)
    except:
        ajouter_progress("Legend not found, stopping the program")
        return
    ajouter_progress("Legend read.")
    ajouter_progress("Matching the legend with the pdf...")
    try:
        read_colonne(chemin_image)
        ajouter_progress("Program done.")
    except:
        ajouter_progress("Program unsuccessfull.")
        return;
    ajouter_progress("Updating the graph...")
    modif_chart_image()
    ajouter_progress("Graph updated.")



if uploaded_file is not None:
    """ Code called after modifying the file """
    file_name = uploaded_file.name
    executeProgram(file_name)
