# **Abstract:**

The app helps the tourist to get information about the monument
which stand front it by capture image and recognize it, tourist can
check the weather, also can know the foreign currency against
Egyptian pound, App shows to tourist the most interesting
beautiful places inside Egypt and can also Booking trip to these
places.


# **Neural Network mapping**
**Input layer**

• Take image of shape (200,200,3)

**Hidden layers**

• 3 Boxs 

• Fully connect:

▪ Flatten data into vector

▪ RULE function of 128 neurons

▪ RULE function of 4028 neurons

▪ Dropout (0.5)

**Output layer**

• Using SoftMax for classification

# **Web Service**

Creating rest service to connnected mobile app with tranning model.
