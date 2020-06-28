from keras.datasets import cifar10
import keras.utils as utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

# GETTING IMAGE DATA ############################################################################################
labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
train_labels = utils.to_categorical(y_train)
test_labels = utils.to_categorical(y_test)

# BUILDING THE MODEL ############################################################################################
# INITIALIZE THE MODEL
# Sequential type lets us add layers IN ORDER
model = Sequential()

# ADD THE FIRST LAYER (CONVOLUTIONAL)
# Params:
#   1. filters: essentially size of image; number of inputs??
#   2. kernel_size: dimensions for kernel (typically 3x3)
#   3. input_shape: 3D dimensions of figure
#   4. activation: type  of activation funciton
#   5. padding: 'same' ensures the image doesn't shrink in the convolutional layer
#   6. kernel_constraint: ensures that large numbers aren't being used; scales numbers down in kernel
model.add(Conv2D(filters=32,kernel_size=(3, 3), input_shape=(32, 32, 3), activation='reLu', padding='same',
                 kernel_constraint=maxnorm(3)))

# ADD THE SECOND LAYER (MAX POOLING)
# Params:
#   1. pool_size: finds the mac value in each n x n section of input matrix
model.add(MaxPooling2D(pool_size=(2, 2)))

# BEFORE PUTTING FEATURES INTO DENSE LAYER NEED TO FLATTEN
model.add(Flatten)

# NOW WE CAN PUT THIS INTO A DENSE LAYER
# where the 'thinking' happens
# Params:
#   1. units: number of neurons; directly proportional to how long you want model to train
#   2. activation
#   3. kernel_constraint
model.add(Dense(units=512, activation='reLu', kernel_constraint=maxnorm(3)))

# NOW ADD DROUPOUT LAYER
# a dropout layer's funciton is to kill neurons in the model; theory is ...
# ... that if model can be accurate with less neurons than it can be even more accurate with more ... obvi
# Params:
#   1. rate: what percantage of neurons you want to drop in decimal form
model.add(Dropout(rate=0.5))

# ADD FINAL DENSE LAYER
# this layer is the final layer of the model so using softmax activation
# where we will generate probabilities of different categories being correct
# Params:
#   1. units: however many categories you are returning
#   2. activation
model.add(Dense(units=10, activation='softmax'))

# COMPILING MODEL #################################################################################################
model.compile(optimizer=SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10, batch_size=32)


