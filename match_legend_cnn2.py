import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
from skimage import exposure, transform, util
import random
import tensorflow as tf
import keras
from keras import layers, models, Input, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from math import floor
import matplotlib.pyplot as plt


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


labels_index_101 = np.asarray([(1, 'shale'), (2, 'clay'), (3, 'silt'), (4, 'sand'), (8, 'marl'), (9, 'chaik'), 
          (10, 'dolomite'), (11, 'limestone'), (12, 'statoil'), (14, 'salt'), 
          (15, 'tuff'), (16, 'coal'), (17, 'anhydrite'), (24, 'shell'), 
          (25, 'fossils'), (26, 'wood'), (27, 'forams'), (32, 'chert'), (33, 'siderite'), (34, 'glauconite'), 
          (35, 'pyrite'),  (42, 'map'), (43, 'fractures'), (44, 'burrotts'), (45, 'stylolites'),
            (46, 'mica'), (47,"notfound")])

column_l_101 = np.array([(0,"notfound"), (1,"notfound"), (2, "notfound"), (3, "notfound"), (4,"notfound"),(5,"notfound"), (6, "notfound"),(7, "notfound"),(8, "notfound"),(9, "notfound"),(10, "notfound"),(11, "notfound"), (12,"clay"),(13,"clay"),(14,"clay"),(15,"clay"),(16,"clay"),(17,"clay"),
               (18,"sand"),(19,"sand"),(20,"sand"),(21,"sand"),(22,"sand"),(23,"sand"),(24,"sand"),(25,"sand"),(26,"clay"),(27,"clay"),(28,"clay"),(29,"sand"),(30,"sand"),(31,"sand"),(32,"sand"),(33,"sand"),(34,"sand"),(35,"clay"),
               (36,"clay"),(37,"clay"),(38,"clay"),(39,"clay"),(40,"clay"),(41,"clay"),(42,"clay"),(43,"clay"),(44,"clay"),(45,"clay"),(46,"clay"),(47,"clay"),(48,"clay"),(49,"clay"),(50,"clay"),(51,"clay"),(52,"clay"),(53,"clay"),
               (54,"clay"),(55,"clay"),(56,"clay"),(57,"clay"),(58,"clay"),(59,"clay"),(60,"clay"),(61,"clay"),(62,"clay"),(63,"clay"),(64,"clay"),(65,"clay"),(66,"clay"),(67,"clay"),(68,"clay"),(69,"clay"),(70,"clay"),(71,"clay"),
               (72,"clay"),(73,"clay"),(74,"clay"),(75,"clay"),(76,"clay"),(77,"clay"),(78,"clay"),(79,"clay"),(80,"clay"),(81,"clay"),(82,"clay"),(83,"clay"),(84,"clay"),(85,"clay"),(86,"clay"),(87,"clay"),(88,"clay"),(89,"clay"),
               (90,"clay"),(91,"clay"),(92,"clay"),(93,"clay"),(94,"sand"),(95,"clay"),(96,"clay"),(97,"clay"),(98,"clay"),(99,"clay"),(100,"clay"),(101,"sand"),(102,"clay"),(103,"clay"),(104,"clay"),(105,"clay"),(106,"clay"),
               (107,"clay"),(108,"clay"),(109,"clay"),(110,"clay"),(111,"clay"),(112,"clay"),(113,"clay"),(114,"clay"),(115,"clay"),(116,"clay"),(117,"clay"),(118,"clay"),(119,"clay"),(120,"clay"),(121,"clay"),(122,"clay"),(123,"clay"),
               (124,"clay"),(125,"clay"),(126,"clay"),(127,"clay"),(128,"clay"),(129,"sand"),(130,"sand"),(131,"sand"),(132,"sand"),(133,"sand"),(134,"sand"),(135,"sand"),(136,"sand"),(137,"sand"),(138,"sand"),(139,"sand"),(140,"sand"),
               (141,"sand"),(142,"sand"),(143,"sand"),(144,"sand"),(145,"sand"),(146,"sand"),(147,"sand"),(148,"sand"),(149,"sand"),(150,"sand"),(151,"sand"),(152,"sand"),(153,"sand"),(154,"sand"),(155,"sand"),(156,"sand"),(157,"sand"),
               (158,"sand"),(159,"sand"),(160,"sand"),(161,"sand"),(162,"sand"),(163,"sand"),(164,"sand"),(165,"sand"),(166,"sand"),(167,"sand"),(168,"sand"),(169,"sand"),(170,"clay"),(171,"clay"),(172,"clay"),(173,"clay"),(174,"clay"),
               (175,"clay"),(176,"clay"),(177,"clay"),(178,"clay"),(179,"clay"),(180,"clay"),(181,"clay"),(182,"clay"),(183,"clay"),(184,"clay"),(185,"clay"),(186,"clay"),(187,"clay"),(188,"clay"),(189,"clay"),(190,"clay"),(191,"clay"),
               (192,"clay"),(193,"clay"),(194,"clay"),(195,"clay"),(196,"clay"),(197,"clay"),(198,"clay"),(199,"clay"),(200,"clay"),(201,"clay"),(202,"clay"),(203,"clay"),(204,"clay"),(205,"clay"),(206,"clay"),(207,"clay"),(208,"clay"),
               (209,"clay"),(210,"clay"),(211,"clay"),(212,"clay"),(213,"clay"),(214,"clay"),(215,"clay"),(216,"clay"),(217,"clay"),(218,"clay"),(219,"clay"),(220,"clay"),(221,"clay"),(222,"clay"),(223,"clay"),(224,"clay"),(225,"clay"),
               (226,"clay"),(227,"clay"),(228,"clay"),(229,"clay"),(230,"clay"),(231,"clay"),(232,"clay"),(233,"clay"),(234,"clay"),(235,"clay"),(236,"clay"),(237,"clay"),(238,"clay"),(239,"clay"),(240,"clay"),(241,"clay"),(242,"clay"),
               (243,"clay"),(244,"clay"),(245,"clay"),(246,"clay"),(247,"clay"),(248,"clay"),(249,"clay"),(250,"clay"),(251,"clay"),(252,"clay"),(253,"clay"),(254,"clay"),(255,"clay"),(256,"clay"),(257,"clay"),(258,"clay"),(259,"clay"),
               (260,"clay"),(261,"clay"),(262,"clay"),(263,"clay"),(264,"clay"),(265,"clay"),(266,"clay"),(267,"clay"),(268,"clay"),(269,"clay"),(270,"clay"),(271,"clay"),(272,"clay"),(273,"clay"),(274,"clay"),(275,"clay"),(276,"clay"),
               (277,"clay"),(278,"clay"),(279,"clay"),(280,"clay"),(281,"clay"),(282,"clay"),(283,"clay"),(284,"clay"),(285,"clay"),(286,"clay"),(287,"clay"),(288,"clay"),(289,"clay"),(290,"clay"),(291,"clay"),(292,"clay"),(293,"clay"),
                (294,"clay"),(295,"clay"),(296,"clay"),(297,"clay"),(298,"clay"),(299,"clay"),(300,"clay"),(301,"clay"),(302,"clay"),(303,"clay"),(304,"clay"),(305,"clay"),(306,"clay"),(307,"clay"),(308,"clay"),(309,"clay"),(310,"clay"),
                (311,"clay"),(312,"clay"),(313,"clay"),(314,"clay"),(315,"clay"),(316,"clay"),(317,"clay"),(318,"clay"),(319,"clay"),(320,"clay"),(321,"clay"),(322,"clay"),(323,"clay"),(324,"clay"),(325,"clay"),(326,"clay"),(327,"clay"),
                (328,"clay"),(329,"clay"),(330,"clay"),(331,"clay"),(332,"clay"),(333,"clay"),(334,"clay"),(335,"clay"),(336,"clay"),(337,"clay"),(338,"clay"),(339,"clay"),(340,"clay"),(341,"clay"),(342,"clay"),(343,"clay"),(344,"clay"),
                (345,"clay"),(346,"clay"),(347,"clay"),(348,"clay"),(349,"clay"),(350,"clay"),(351,"clay"),(352,"clay"),(353,"clay"),(354,"clay"),(355,"clay"),(356,"clay"),(357,"clay"),(358,"clay"),(359,"clay"),(360,"clay"),(361,"clay"),
                (362,"clay"),(363,"clay"),(364,"clay"),(365,"clay"),(366,"clay"),(367,"clay"),(368,"clay"),(369,"clay"),(370,"clay"),(371,"clay"),(372,"clay"),(373,"clay"),(374,"clay"),(375,"clay"),(376,"sand"),(377,"sand"),(378,"sand"),
                (379,"sand"),(380,"sand"),(381,"sand"),(382,"sand"),(383,"sand"),(384,"sand"),(385,"sand"),(386,"sand"),(387,"sand"),(388,"sand"),(389,"sand"),(390,"sand"),(391,"sand"),(392,"sand"),(393,"limestone"),(394,"limestone"),(395,"limestone"),
                (396,"limestone"),(397,"limestone"),(398,"limestone"),(399,"limestone"),(400,"limestone"),(401,"limestone"),(402,"limestone"),(403,"limestone"),(404,"limestone"),(405,"limestone"),(406,"limestone"),(407,"limestone"),(408,"limestone"),(409,"limestone"),(410,"limestone"),(411,"limestone"),
                (412,"limestone"),(413,"limestone"),(414,"limestone"),(415,"limestone"),(416,"limestone"),(417,"limestone"),(418,"marl"),(419,"marl"),(420,"marl"),(421,"limestone"),(422,"limestone"),(423,"silt"),(424,"silt"),(425,"shale"),(426,"shale"),(427,"shale"),(428,"shale"),(429,"shale"),
                (430,"sand"),(431,"sand"),(432,"sand"),(433,"sand"),(434,"sand"),(435,"sand"),(436,"sand"),(437,"sand"),(438,"sand"),(439,"sand"),(440,"sand"),(441,"sand"),(442,"sand"),(443,"sand"),(444,"sand"),(445,"sand"),(446,"sand"),
                (447,"sand"),(448,"sand"),(449,"sand"),(450,"sand"),(451,"shale"),(452,"shale"),(453,"shale"),(454,"shale"),(455,"shale"),(456,"shale"),(457,"shale"),(458,"shale"),(459,"shale"),(460,"shale"),(461,"shale"),(462,"shale"),(463,"shale"),
                (464,"shale"),(465,"shale"),(466,"shale"),(467,"shale"),(468,"shale"),(469,"shale"),(470,"shale"),(471,"shale"),(472,"shale"),(473,"shale"),(474,"shale"),(475,"shale"),(476,"shale"),(477,"shale"),(478,"shale"),(479,"shale"),
                (480,"shale"),(481,"shale"),(482,"shale"),(483,"shale"),(484,"shale"),(485,"shale"),(486,"shale"),(487,"anhydrite"),(488,"anhydrite"),(489,"anhydrite"),(490,"dolomite"),(491,"dolomite"),(492,"sand"),(493,"sand"),(494,"sand"),(495,"sand"),(496,"sand")])



def partition_lith_column(img, step, size):
    """
    Partition lith column into chunks of height "size" with a step of "step"
    """
    partitions = []
    for i in range(0, img.shape[0], step):
        if i+size < img.shape[0]:
            partitions.append(img[i:i+size, :])
    return partitions


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        print(str(filename))
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(filename.split(".")[0])
          
    return images,labels

def resize_images(images, target_size=(130, 90)):
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        resized_images.append(img_resized)
    return resized_images

def images_to_arrays(images):
    image_arrays = []
    for img in images:
        img_array = np.asarray(img)
        image_arrays.append(img_array)
    return image_arrays

def process_images(images):
    processed_images = []
    for img in images:
        #Convert to grayscale
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #normalize
        img_processed = img_processed / 255
        processed_images.append(img_processed)
    return processed_images

labels_index_183 = np.asarray([
    (1,"clay"), (2,"silt"), (3,"sand"), (4,"uwu"),(9,"dolomite"),(10,"chalk"), (11,"limestone"), (12,"shale"),(14,"anhydrite"),(15,"salt"),(16,"tuff"),(17,"marl"),(22,"shell"), (23,"fossils"),(24,"wood"),
    (25,"coal"),(29,"glauconite"),(30,"pyrite"),(31,"burrows"),(32,"forams"),(39,"chert"),(40,"uwu2"),(41,"uwu3"),(42,"mica")
])

column_l_183 = np.asarray(["clay","clay","clay","sand","sand","sand","clay","clay","sand","sand","sand", "sand", "sand","clay","clay","sand","sand","clay","clay"])

#data augmentation

#only works for 3 channel images
def random_brightness(image, min_value=0.5, max_value=1.5):
    value = np.random.uniform(min_value, max_value)
    hsv = cv2.cvtColor(image, cv2.COLOR_HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_contrast(image, min_value=0.5, max_value=1.5):
    alpha = np.random.uniform(min_value, max_value)
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return new_image

def random_rotation(image, max_angle=15):
    angle = random.uniform(-max_angle,max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated

def random_flip(image, p=0.5):
    if np.random.rand() < p:
        image = cv2.flip(image, 1) # Horizontal flip
    return image

def random_noise(image):
    return util.random_noise(image, mode='gaussian', seed=None, clip=True, mean=0, var=0.01)

def random_scale(image, min_scale=0.8, max_scale=1.2):
    scale = np.random.uniform(min_scale, max_scale)
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    if scale > 1:  # If the image was upscaled, crop it to the original size
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        cropped = resized[y_offset:y_offset+h, x_offset:x_offset+w]
        return cropped
    else:  # If the image was downscaled, pad it to the original size with reflection
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        padded = cv2.copyMakeBorder(resized, y_offset, h - new_h - y_offset, x_offset, w - new_w - x_offset, cv2.BORDER_REFLECT_101)
        return padded
    
def apply_augmentation(image):
    #augmented_image = random_brightness(image)
    augmented_image = random_contrast(image,0.8,1.2)
    augmented_image = random_rotation(augmented_image)
    augmented_image = random_flip(augmented_image)
    #augmented_image = random_noise(augmented_image)
    augmented_image = random_scale(augmented_image)
    return (augmented_image * 255).astype(np.uint8)

def augment_dataset(images, n_augmentations=3):
    augmented_images = []
    for image in images:
        augmented_images.append(image * 255)
        for _ in range(n_augmentations):
            augmented_image = apply_augmentation(image)
            augmented_images.append(augmented_image)
    return augmented_images



def prepare_data(legend_folder, column_folder):
    legend_images, legend_labels = load_images_from_folder(legend_folder)
    column_images, column_labels = load_images_from_folder(column_folder)

    combined_list = zip(np.array(column_labels).astype(int), column_images)
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

    sorted_column_labels, sorted_column_images = zip(*sorted_combined_list)
    # print(str(column_l_101[:,1]))
    column_labels = np.concatenate((np.array(["notfound","notfound"]), column_l_101[:,1]))

    legend_images_resized = resize_images(legend_images, (130,90))
    column_images_resized = resize_images(sorted_column_images, (1000,200))

    #process images
    legend_images_processed = process_images(legend_images_resized)
    column_images_processed = process_images(column_images_resized)
    # print("Legend images and labels:")
    # for i, img in enumerate(legend_images_processed):
    #     print(f"Label: {legend_labels[i]}")
    #     plt.imshow(img.squeeze(), cmap="gray")
    #     plt.show()


    # Debug print column images and labels
    # print("Column images and labels:")
    # for i, img in enumerate(column_images_processed):
    #     print(f"Label: {column_labels[i]}")
    #     plt.imshow(img.squeeze(), cmap="gray")
    #     plt.show()


    legend_arrays = images_to_arrays(legend_images_processed)
    column_arrays = images_to_arrays(column_images_processed)

    print("legend label = " + str(np.unique(legend_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(legend_labels)}

    column_labels = column_l_101[:,1]
    print("column label = " + str(np.unique(column_labels)))

    legend_labels.append("notfound")
    # Convert labels to integers
    unique_legend_labels = sorted(list(set(legend_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_legend_labels)}

    legend_labels_int = [label_to_idx[label] for label in legend_labels]
    column_labels_int = [label_to_idx[label] for label in column_labels]

    #print the label and their index
    for i in range(len(unique_legend_labels)):
        print("Label: " + str(unique_legend_labels[i]) + " Index: " + str(i))
    data = {
        "legend_images": np.array(legend_arrays).astype(np.float32),
        "legend_labels": np.array(legend_labels_int),
        "column_images": np.array(column_arrays).astype(np.float32),
        "column_labels": np.array(column_labels_int)
    }
    
    print(str(data["legend_labels"]))
    print(str(data["column_labels"]))
    return data

def prepare_for_predict(legend_folder,column_folder):
    legend_images, legend_labels = load_images_from_folder(legend_folder)
    column_images, _ = load_images_from_folder(column_folder) # the column labels are not known if we want to predict

    legend_images_resized = resize_images(legend_images, (130,90))
    column_images_resized = resize_images(column_images, (1000,200))

    #process images
    legend_images_processed = process_images(legend_images_resized)
    column_images_processed = process_images(column_images_resized)

    legend_arrays = images_to_arrays(legend_images_processed)
    column_arrays = images_to_arrays(column_images_processed)

    unique_legend_labels = sorted(list(set(legend_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_legend_labels)}
    legend_labels_int = [label_to_idx[label] for label in legend_labels]

    data = {
        "legend_images": np.array(legend_arrays).astype(np.float32),
        "legend_labels": np.array(legend_labels_int),
        "column_images": np.array(column_arrays).astype(np.float32),
    }

    return data





def create_cnn_branch(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    return model

def create_siamese_network(column_input_shape, legend_input_shape, num_classes):
    column_branch = create_cnn_branch(column_input_shape)
    legend_branch = create_cnn_branch(legend_input_shape)

    column_input = Input(column_input_shape)
    legend_input = Input(legend_input_shape)

    column_features = column_branch(column_input)
    legend_features = legend_branch(legend_input)

    # Merge the feature vectors using L1 distance
    l1_distance_layer = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([column_features, legend_features])

    # Classification layer
    output = layers.Dense(1, activation='sigmoid')(l1_distance)


    siamese_network = models.Model(inputs=[column_input, legend_input], outputs=output)

    siamese_network.compile(optimizer='adam',
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy'])

    return siamese_network

def create_pairs(column_images, column_images_legend, legend_images, legend_images_legend):
    num_column = len(column_images)
    num_legend = len(legend_images)
    pairs = []
    result = []

    for i in range(num_column):
        for j in range(num_legend):
            pairs.append([column_images[i], legend_images[j]])
            result.append(column_images_legend[i] == legend_images_legend[j])
            # print("legend: " + str(legend_images_legend[j]) + " column: " + str(column_images_legend[i]))
            # if column_images_legend[i] == legend_images_legend[j]:
            #     print("Match")
            # else:
            #     print("No match")
            # plt.imshow(column_images[i].squeeze(), cmap="gray")
            # plt.show()

    
    return pairs, result



# legend_folder = 'cropped/'
# column_folder = 'partitions/'

# data = prepare_data(legend_folder, column_folder)


# print("Column images : " + str(len(data['column_images'])))
# print("Column labels : " + str(len(data['column_labels'])))

# data['column_labels'] = np.append(data['column_labels'],14)
# data['column_labels'] = np.append(data['column_labels'],14)

# column_images_train, column_images_val, column_images_train_legend, column_images_val_legend = train_test_split(data["column_images"][:200], data['column_labels'][:200], test_size=0.2, random_state=52)


# pairs_train, result_train = create_pairs(column_images_train, column_images_train_legend, data['legend_images'], data['legend_labels']) 
# pairs_val, result_val = create_pairs(column_images_val, column_images_val_legend, data['legend_images'], data['legend_labels'])

#Verify that the column images and labels correspond
#Debug print column images and labels
# print("Column images and labels:")
# for i, img in enumerate(column_images_train):
#     print("label = " + str(column_images_train_legend[i]))
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.show()

#Verify that the legend images and labels correspond
#Debug print legend images and labels
# print("Legend images and labels:")
# for i, img in enumerate(data["legend_images"]):
#     print("label = " + str(data["legend_labels"][i]))
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.show()




#for i in range(len(legend_images)):
#    cv2.imwrite("legend_images/legend_" + str(data['legend']['labels'][i]) + ".png", legend_images[i]*255)
# column_input_shape = (200, 1000, 1)  
# legend_input_shape = (90, 130, 1)  

# num_classes = len(np.unique(data["legend_labels"]))
# print("num_classes = " + str(num_classes))



# siamese_network = create_siamese_network(column_input_shape, legend_input_shape, num_classes)


# print(str(siamese_network.summary()))

# # Convert the data to numpy arrays
# pairs_train = np.array(pairs_train)
# pairs_val = np.array(pairs_val)

# print("Reshaping the pairs arrays")
# print("Reshaping column images")
# train_column_images = np.stack(pairs_train[:, 0], axis=0)
# print("Reshaped column images")
# train_legend_images = np.stack(pairs_train[:, 1], axis=0)
# print("Reshaped legend images")
# val_column_images = np.stack(pairs_val[:, 0], axis=0)
# print("Reshaped column images")
# val_legend_images = np.stack(pairs_val[:, 1], axis=0)
# print("Reshaped legend images")
# result_train = np.array(result_train)
# result_val = np.array(result_val)

# print("Reshaped the pairs arrays")
# print("train_column_images.shape = " + str(train_column_images.shape))
# print("train_legend_images.shape = " + str(train_legend_images.shape))
# print("val_column_images.shape = " + str(val_column_images.shape))
# print("val_legend_images.shape = " + str(val_legend_images.shape))
# print("result_train.shape = " + str(result_train.shape))
# print("result_val.shape = " + str(result_val.shape))




# [print(i.shape, i.dtype) for i in siamese_network.inputs]
# [print(o.shape, o.dtype) for o in siamese_network.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in siamese_network.layers]





# print("done")
# #train the network
# history = siamese_network.fit(
#     [train_column_images, train_legend_images],
#     result_train,
#     batch_size=8,
#     epochs=10,
#     validation_data=([val_column_images, val_legend_images], result_val)
# )

# print("done")

# #save the model to disk
# print("[INFO] serializing network...")
# siamese_network.save("siamese_network.h5")

#load the model from disk
print("[INFO] loading network...")
model = keras.models.load_model("siamese_network.h5")

# evaluate the network
# print("[INFO] evaluating network...")
#results = model.evaluate([data["column_images"][:200], data['legend_images'][:200]], result_val)

# num_column = len(data['column_images'][:200])
# num_legend = len(data['legend_images'][:200])
# pairs = []
# result = []

# for i in range(num_column):
#     for j in range(num_legend):
#         if data['column_labels'][i] == data['legend_labels'][j]:
#             pairs.append([data['column_images'][i], data['legend_images'][j]])
#             result.append(data['column_labels'][i] == data['legend_labels'][j])
#         # print("legend: " + str(legend_images_legend[j]) + " column: " + str(column_images_legend[i]))
#         # if column_images_legend[i] == legend_images_legend[j]:
#         #     print("Match")
#         # else:
#         #     print("No match")
#         # plt.imshow(column_images[i].squeeze(), cmap="gray")
#         # plt.show()
# pairs_test = np.array(pairs)
# print(str(pairs_test.shape))
# test_column_images = np.stack(pairs_test[:, 0], axis=0)
# test_legend_images = np.stack(pairs_test[:, 1], axis=0)
# result_test = np.array(result)

# results = model.predict([test_column_images, test_legend_images])
# print("results = " + str(results))
# print("result_test = " + str(result_test))



# print("predictions = " + str(predictions))

# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 10), history.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 10), history.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 10), history.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 10), history.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")
# 


# res_195_traficote = cv2.imread("res_195_traficote.jpg")
# imgs = partition_lith_column(res_195_traficote, 100, 200)
# for i in range(len(imgs)):
#     cv2.imwrite("partitions2/" + str(i) + ".jpg", imgs[i])

data = prepare_for_predict("cropped2","partitions2")

print("data['column_images'].shape = " + str(data['column_images'].shape))
print("data['legend_images'].shape = " + str(data['legend_images'].shape))
print("data['legend_labels'].shape = " + str(data['legend_labels'].shape))

pairs = []
result = []
for i in range(100):
    for j in range(len(data['legend_images'])):
        pairs.append([data['column_images'][i], data['legend_images'][j]])

pairs_test = np.array(pairs)
print(str(pairs_test.shape))
test_column_images = np.stack(pairs_test[:, 0], axis=0)
test_legend_images = np.stack(pairs_test[:, 1], axis=0)

results = model.predict([test_column_images, test_legend_images])

for i in range(100):
    max = 0
    max_index = 0
    for j in range(len(data['legend_images'])):
        if results[i*len(data['legend_images']) + j] > max:
            max = results[i*len(data['legend_images']) + j]
            max_index = j

    print("column " + str(i) + " is " + str(data['legend_labels'][max_index]))




