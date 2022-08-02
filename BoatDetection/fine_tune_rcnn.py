from keras import Sequential
from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import pickle
import os

import config


print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []


for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

print('imagePaths: ', len(imagePaths))
print('data: ', len(data))
print('labels: ', len(labels))

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=config.train_test_split_proportion,
    stratify=labels,
    # random_state=42
)
print('trainX: ', trainX.shape, 'testX: ', testX.shape, 'trainY: ', trainY.shape, 'testY: ', testY.shape)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    # rescale=1/255.,
    rotation_range=config.aug_rotation_range,
    zoom_range=config.aug_zoom_range,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)


def raw_model_cnn():
    model = Sequential()
    model.add(Conv2D(4, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)))
    model.add(AveragePooling2D(4, 4))
    model.add(Dropout(0.2))

    model.add(Conv2D(4, (5, 5), padding="same", activation="relu"))
    model.add(AveragePooling2D(4, 4))
    # model.add(Dropout(0.2))

    model.add(Conv2D(4, (5, 5), padding="same", activation="relu"))
    model.add(AveragePooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))

    model.summary()
    model.compile(optimizer=Adam(lr=config.INIT_LR), loss="binary_crossentropy", metrics=['accuracy'])
    return model


def raw_model_MobileNetV2():
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=config.INPUT_DIMS_3D))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    model.summary()
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=config.INIT_LR), metrics=["accuracy"])

    return model


def raw_model_Inception():
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.optimizers import RMSprop

    base_model = InceptionV3(input_shape=config.INPUT_DIMS_3D, include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    headModel = Flatten()(base_model.output)
    headModel = Dense(128, activation='relu')(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(2, activation='sigmoid')(headModel)
    model = Model(base_model.input, headModel)

    model.summary()
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    return model


def raw_model_EfficientNetB0():
    import efficientnet.keras as efn

    base_model = efn.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    headModel = Flatten()(base_model.output)
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)
    model = Model(input=base_model.input, output=headModel)

    model.summary()
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model, path_to_save):
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=config.BS, shuffle=True),
        steps_per_epoch=len(trainX) // config.BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // config.BS,
        epochs=config.EPOCHS,
        shuffle=True
    )
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=config.BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

    print("[INFO] saving mask detector model...")
    model.save(
        path_to_save,
        # save_format="h5"
    )
    print("[INFO] saving label encoder...")
    f = open(path_to_save+'/label_encoder_boat.pickle', "wb")
    f.write(pickle.dumps(lb))
    f.close()


model = raw_model_MobileNetV2()
fit_model(model, config.RESULT_MODEL_PATH_MobileNet)


# model = cnn_raw_model()
# fit_model(model, config.RESULT_MODEL_PATH_CNN)


# model = raw_model_Inception()
# fit_model(model, config.RESULT_MODEL_PATH_Inception)

# model = raw_model_EfficientNetB0()
# fit_model(model, config.RESULT_MODEL_PATH_EfficientNetB0)
