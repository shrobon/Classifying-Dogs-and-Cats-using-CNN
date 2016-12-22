import numpy as np
import argparse
import __future__
from imutils import paths
import cv2,random
from matplotlib import pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import model_from_json # will be used to save the weights of the model 
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

print("")
print("Learning Deep Neural Networks from SCRATCH")
print("===========================================")

#### These are my input image parameters ####
ROWS = 64
COLS = 64
CHANNELS = 3
##############################################

ap = argparse.ArgumentParser()
ap.add_argument("-train","--train",required=True,help="path to training images")
args = vars(ap.parse_args())

#image paths gives the complete path of the images in the directory
imagePaths = list(paths.list_images(args["train"]))
cats =[i for i in imagePaths if 'cat' in i]
dogs =[i for i in imagePaths if 'dog' in i]


train_images = dogs[:10000] + cats[:10000]
random.shuffle(train_images)
print("Total Number of training Images :  %d"%(len(train_images)))

test_images = dogs[10001:] + cats[10001:]
random.shuffle(test_images)
print("Total Number of training Images : %d"%len(test_images))


##Binary Classification so CAT = 0 , DOG = 1
labels =[]
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)


testlabels =[]
for i in test_images:
    if 'dog' in i:
        testlabels.append(1)
    else:
        testlabels.append(0)

labels = np.asarray(labels);
testlabels = np.asarray(testlabels);



#helper function to read the image and resize it accordingly
#There is a need to understand the interpolation method
def read_image(file_path):
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((ROWS,COLS))
    img = np.asarray(img)    
    return img
    #return Image.fromarray(img.astype('uint8'), 'RGB')
    #img = cv2.imread(file_path)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #return cv2.resize(img,(ROWS,COLS),interpolation = cv2.INTER_AREA)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    print("Shape of data is ::")
    print(np.shape(data))
    
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.transpose()
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    return data

train = prep_data(train_images)
test = prep_data(test_images)
print("Shape of train is ::")
print(np.shape(train))
print("Shape of test is ::")
print(np.shape(test))


objective = 'binary_crossentropy'
optimizer = 'Adadelta'
def catdog(optimizer='Adam',activation='relu'):
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256,init='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Dropout(0.2))
    
    model.add(Dense(256,init='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    model.add(Dropout(0.2))

    model.add(Dense(1,init='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))





model = catdog()
model.summary() 
plot(model, to_file='model.png') 

nb_epoch = 200
batch_size = 32
#early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

def run_catdog():
    print("[INFO] Model is being Trained...")
    datagen = ImageDataGenerator(
        rotation_range=90,
        fill_mode='wrap',
        horizontal_flip=True)
    datagen.fit(train)
    history = LossHistory()
    model.fit_generator((datagen.flow(train,labels, batch_size=batch_size)),samples_per_epoch=len(train), nb_epoch=nb_epoch,callbacks=[history],verbose=1,validation_data=(test, testlabels))
    return history




history = run_catdog()
loss = history.losses
val_loss = history.val_losses
#Displaying the Validation Loss and Training Loss with the number of epochs
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()




print("[INFO] Saving the model and weights...")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




