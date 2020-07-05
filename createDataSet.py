# Import some important package
from random import shuffle
import glob
import numpy as np
import h5py
import cv2

# Set where save dataset
shuffle_data = True
hdf5_path = '/path.hdf5'

# Get the images
pyramids_train_path = 'path of directory/*.jpg'

# Set the classes of dataset
addrs = glob.glob(pyramids_train_path)
labels = [0 if 'pyramids' in addr else 1 if 'abusimbel' in addr else 2 if 'sphinx' in addr else 3
for addr in addrs]

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Set the size of train/dev/test set
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))
                     ]
val_addrs = addrs[int(0.8 * len(addrs)):int(0.9 * len(addrs))]
val_labels = labels[int(0.8 * len(labels)):int(0.9 * len(labels))]

test_addrs = addrs[int(0.9*len(addrs)):]
test_labels = labels[int(0.9*len(labels)):]

# Choose the order of the data tf or th
data_order = 'tf'

if data_order == 'th':
    train_shape = (len(train_addrs), 3, 200, 200)
    val_shape = (len(val_addrs), 3, 200, 200)
    test_shape = (len(test_addrs), 3, 200, 200)
    
elif data_order == 'tf':
    train_shape = (len(train_addrs), 200, 200, 3)
    val_shape = (len(val_addrs), 200, 200, 3)
    test_shape = (len(test_addrs), 200, 200, 3)

# Organize your dataset
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, dtype=np.uint8)
hdf5_file.create_dataset("val_img", val_shape, dtype=np.uint8)
hdf5_file.create_dataset("test_img", test_shape, dtype=np.uint8)

hdf5_file.create_dataset("train_mean", train_shape[1:], dtype=np.float32)

hdf5_file.create_dataset("train_labels", (len(train_addrs),), dtype=np.uint8)
hdf5_file["train_labels"][...] = train_labels

hdf5_file.create_dataset("val_labels", (len(val_addrs),), dtype=np.uint8)
hdf5_file["val_labels"][...] = val_labels

hdf5_file.create_dataset("test_labels", (len(test_addrs),), dtype=np.uint8)
hdf5_file["test_labels"][...] = test_labels

mean = np.zeros(train_shape[1:], dtype=np.float32)

for i in range(len(train_addrs)):
    if i % 30 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
    
for i in range (len(val_addrs)):
    if i % 30 == 0 and i > 1:
        print('Validation data : {}/{}'.format(i, len(val_addrs)))
       
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
   if data_order == 'th':
     img = np.rollaxis(img, 2)
       
hdf5_file["val_img"][i, ...] = img[None]

for i in range (len(test_addrs)):
    if i % 30 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if data_order == 'th':
        img = np.rollaxis(img, 2)
        
    hdf5_file["test_img"][i, ...] = img[None]
    
hdf5_file["train_mean"][...] = mean
hdf5_file.close()
