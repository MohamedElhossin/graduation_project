# Import some important package
from random import shuffle
import glob
import numpy as np
import h5py
import cv2

# Load the data set
hdf5_file = h5py.File ('/path of dataset.hdf5','r')

# Prepare the data set for training
train_data = hdf5_file['train_img']
X_train = np.array(train_data)
X_train = X_train / 255.

train_label = hdf5_file['train_labels']
Y_train = np.array(train_label)
Y_train = np.reshape(Y_train, (train_label.shape[0], 1))

yt = Y_train.T
Y_train_label = np.zeros((yt.shape[1], 4))
Y_train_label[np.arange(yt.shape[1]), yt] = 1

val_data = hdf5_file['val_img']
X_val = np.array(val_data)
X_val = X_val / 255.

val_labels = hdf5_file['val_labels']
Y_val = np.array(val_labels)
Y_val = np.reshape(Y_val, (val_labels.shape[0], 1))

a = Y_val.T
Y_val_label = np.zeros((a.shape[1], 4))
Y_val_label[np.arange(a.shape[1]), a] = 1

# Set the structure of neural network
input_shape = X_train.shape[1:]
X_input = Input(input_shape)
model = ZeroPadding2D((3, 3))(X_input)

#Box (1)
model = Conv2D(filters=32, kernel_size=(3, 3), name='block1_conv1')(model)
model = Activation('relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), name='block1_conv2')(model)
model = Activation('relu')(model)
model = MaxPooling2D(pool_size=(2, 2), name='block1_poool')(model)
#Box (2)
model = Conv2D(filters=64, kernel_size=(3, 3), name='block2_conv1')(model)
model = Activation('relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), name='block2_conv2')(model)
model = Activation('relu')(model)
model = MaxPooling2D(pool_size=(2, 2), name='block2_poool')(model)
#Box (3)
model = Conv2D(filters=128, kernel_size=(3, 3), name='block3_conv1')(model)
model = Activation('relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), name='block3_conv2')(model)
model = Activation('relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), name='block3_conv3')(model)
model = Activation('relu')(model)
model = MaxPooling2D(pool_size=(2, 2), name='block3_poool')(model)
#Fully connect 
model = Flatten()(model)
model = Dense(128, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
# Output layer
model = Dense(4, activation='softmax', name='fc3',
kernel_initializer=glorot_uniform(seed=0))(model)
model = Model(inputs=X_input, outputs=model, name='mymodel')74

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

# Fit the dataset into model for training
model.fit(X_train, Y_train_label,epochs=20, batch_size=32)

# Evaluate model
model.evaluate(X_val, Y_val_label)

# Save model
model.save('/path of saving model.h5')
