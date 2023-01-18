from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
import numpy as np
import os
import random
import gc


# Get images
files = []
# Create batches of 500 images and recursive train on them
for filename in os.listdir('./TrainData/test'):
    files.append(filename)
number_of_batches = int(len(files)/500)
files = np.array(files)
batches = np.array_split(files, number_of_batches + 1)
print("Splitted dataset into", number_of_batches + 1, " batches.")
del number_of_batches
del files
index = 1;

# Generate training data

batches = np.flipud(batches)


for batch in batches:
    print("Batch ", index)
    X=[]
    for filename in batch:
        image = load_img('./TrainData/test/' + filename)
        image = image.resize((512,512))
        image_as_array = img_to_array(image)
        image_as_array = np.array(image_as_array, dtype=float)
        X.append(image_as_array)
        del image
        del image_as_array


    X = np.array(X, dtype=float)
    split = int(0.95*len(X))
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain
    
    model = load_model('model.h5')

    # Image transformer
    datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

    batch_size = 10
    def image_a_b_gen(batch_size):
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

    # Train model
    model.fit(image_a_b_gen(batch_size = 10), epochs=50, steps_per_epoch=15)

    # Save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save("model.h5")


    # Test images
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print("model evaluation ", model.evaluate(Xtest, Ytest, batch_size=batch_size))
    index +=1
    del model
    del Xtrain

