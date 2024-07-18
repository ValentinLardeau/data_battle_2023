import os
import cv2
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


def launch():
    nb_image = 50  # number of picture created for one label to augment the learning base
    p80 = int(nb_image * 0.8)  # 80% of all the pictures
    p20 = nb_image - p80  # 20% of all the pictures

    # neural network trained (soooo obviously good coeff) with great accuracy on the training base (100% lol that's better than great)
    # function well but confused some classes ( like for instance on well 15-9-9 puts clay and limestone together and not really confident with sand too)
    # however will always last about a minute (but that's actually not that long)

    legend_image_path = "./cropped/"
    column = "res_195.jpg"

    legend_images = []
    legend_image_name = []

    # recuperation of the labels
    for file in os.listdir(legend_image_path):
        img_array = cv2.imread(os.path.join(legend_image_path, file))
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        legend_images.append(gray_image)
        legend_image_name.append(file)

    # cut of the column in order to let the pictures go in the ai

    def partition_colonne(img, step, size):
        partition = []
        for i in range(0, img.shape[0], step):
            if (i+size) < img.shape[0]:
                partition.append(img[i:i+size, :])
        return partition

    lith_colonne = cv2.imread(column)
    lith_colonne = cv2.cvtColor(lith_colonne, cv2.COLOR_BGR2GRAY)

    partition = partition_colonne(lith_colonne, 200, 200)

    # shaping the pictures : same shape for all of them and putting their values between 0 and 1 in order to go in the ai
    resize = tf.keras.Sequential([
        layers.Resizing(67, 100),
        layers.Rescaling(1./255),
    ])

    # shaping the column in order to only grab what we want and making the picture as pretty as the labels one
    resize_colonne = tf.keras.Sequential([
        layers.CenterCrop(100, 80),
        layers.Resizing(67, 100),
        layers.Rescaling(1./255),
    ])

    # Function to apply to all pictures to transform them, really efficient for the learning phase
    data_augmented = keras.Sequential([layers.RandomFlip("horizontal"),
                                       layers.RandomContrast(0.2),
                                       layers.RandomRotation(0.05)
                                       ])

    # temporary dataset
    dictio_dataset = {}

    # recuperation of all pictures
    legend = []
    for img in legend_images:
        legend.append(img)

    # creation of a dictionnary shaping as : {0 : [image,...,image], ...., 47 : [image,..., image]} with reshaped images
    for j in range(len(legend)):
        imgs_augmented = []
        img_to_augment = np.expand_dims(legend[j], axis=-1)
        for i in range(nb_image):
            tmp = resize(img_to_augment)
            img_augmented = data_augmented(tmp)
            imgs_augmented.append(img_augmented)
        dictio_dataset[j] = imgs_augmented

    test = dictio_dataset

    # Putting the dictionnary into a pandas dataset because we cannot convert directly into a tf one
    Datatest_panda = pd.DataFrame.from_dict(dictio_dataset)

    # creation of the learning and verification base
    train = (Datatest_panda.head(p80)).to_numpy()
    test = (Datatest_panda.tail(p20)).to_numpy()

    # transforming in list in order to be accepted by tf functions

    train_images = []
    train_label = []
    test_images = []
    test_label = []

    label_liste = [i for i in range(len(legend))]
    print(label_liste)

    for j in range(len(train)):
        for k in range(len(train[j])):
            train_images.append(train[j][k])
            train_label.append(label_liste[k])

    for j in range(len(test)):
        for k in range(len(test[j])):
            test_images.append(test[j][k])
            test_label.append(label_liste[k])

    # change of type to be accepted by tf functions
    train_images = np.asarray(train_images)
    train_label = np.asarray(train_label)
    test_images = np.asarray(test_images)
    test_label = np.asarray(test_label)

    num_classes = len(np.unique(train_label))

    # creation of the model
    # already one existing just new learning base soooooo we can keep the weights
    model = keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # same here, besides the optimizer seems to be the best one
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # training of the first model
    epochs = 10
    # history = model.fit(
    #   train_images,
    #   y=train_label,
    #   validation_data=(test_images,test_label),
    #   epochs=epochs)

    # printing the characteristics of the ia but before having fix the over adjustement sooooo kinda useless
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

    # final enhancement to have the best learning base
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),
                                          layers.RandomContrast(0.2),
                                          layers.RandomRotation(0.05)
                                          ])  # augmentation in order to fix the over adjustement

    model = keras.Sequential([  # correction by abandon
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # creation of the new model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # training of the new ia
    history = model.fit(
        train_images,
        y=train_label,
        validation_data=(test_images, test_label),
        epochs=epochs)

    # printing the characteristics of the good ia
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # recuperation of the image to predict
    preprocessed_image = []
    for img in partition:
        img = np.expand_dims(img, axis=-1)
        resized_img = resize_colonne(img)
        preprocessed_image.append(resized_img)

    # changes of type in order to be accepted by tf functions
    preprocessed_image = np.asanyarray(preprocessed_image)
    nb_image_col = len(preprocessed_image)

    # prediction of all the images to test
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction, axis=1)

    # counting the number of picture in every classes
    list_freq = np.bincount(predicted_label)

    label_list = [i for i in range(len(list_freq))]

    # calcul of the percent
    list_freq_pourcent = []
    for i in range(len(list_freq)):
        list_freq_pourcent.append(list_freq[i]/nb_image_col*100)

    # for i in range(len(preprocessed_image)):
    #     plt.imshow(preprocessed_image[i])
    #     plt.title(predicted_label[i])
    #     plt.show()

    # creation of the pie diagram
    plt.pie(list_freq_pourcent, shadow=True, labels=legend_image_name)
    plt.title("Diagramme circulaire représentant la proportion de chaque roche")
    plt.show()
