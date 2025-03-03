import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

# Update the path to a valid directory on your system
path = '../input/100-bird-species'  # Updated to match the available path seen in the error log
def get_file_number(folder):
    # Count the number of files in a directory
    return sum([len(files) for _, _, files in os.walk(folder)])

# Check if directories exist, print counts only if they do
train_path = os.path.join(path, 'train')
valid_path = os.path.join(path, 'valid')
test_path = os.path.join(path, 'test')

# Print file counts if directories exist
print(get_file_number(train_path) if os.path.exists(train_path) else 'Train directory not found')
print(get_file_number(valid_path) if os.path.exists(valid_path) else 'Valid directory not found')
print(get_file_number(test_path) if os.path.exists(test_path) else 'Test directory not found')

# Update paths to match the new base path
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'valid') 
test_dir = os.path.join(path, 'test')
test_dir = '../input/100-bird-species/test'
batch_size = 32
# instead of using ImageDataGenerator - image_dataset_from_directory() generates a
# tf.data.Dataset (usually faster than the ImageDataGenerator) from image files in a directory
# image_size must be provided, here: reducing intital size of (224, 224) to (128, 128)
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, label_mode='categorical',
image_size=(128, 128), batch_size=batch_size)
valid_dataset = tf.keras.utils.image_dataset_from_directory(valid_dir, label_mode='categorical',
image_size=(128, 128), batch_size=batch_size)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, label_mode='categorical',
image_size=(128, 128), batch_size=batch_size)
label_names = train_dataset.class_names
print(label_names)
plt.figure(figsize=(25, 20))
for images, labels in train_dataset.take(1):
 for i in range(6):
  ax = plt.subplot(1, 6, i+1)
  plt.imshow(images[i].numpy().astype('int'))
  plt.title(train_dataset.class_names[tf.argmax(labels[i])]) # this works to display the labels according to the images
  #plt.title(label_name[labels[i]]) doesnt work
  plt.grid()
  plt.axis('on')

for image_batch, labels_batch in train_dataset:
 print(image_batch.shape)
# normalize the dataset to get (input) values between 0-1
def normalize_data(dataset):
    # To rescale an input in the [0, 255] range to be in the [0, 1] range, you would pass scale=1./255.
    normalization_layer = Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))

# Define the normalized datasets
normalized_training_data = normalize_data(train_dataset)
normalized_validation_data = normalize_data(valid_dataset)
normalized_testing_data = normalize_data(test_dataset)

# Use tf.data.AUTOTUNE for efficient prefetching
AUTOTUNE = tf.data.AUTOTUNE
normalized_training_data = normalized_training_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_validation_data = normalized_validation_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_testing_data = normalized_testing_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_validation_data = normalized_validation_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_testing_data = normalized_testing_data.cache().prefetch(buffer_size=AUTOTUNE)

# Define the model first before using it
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    normalized_training_data,
    validation_data=normalized_validation_data,
    epochs=10
)

plt.figure(figsize=(10,8))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend(['accuracy', 'val_accuracy'])
plt.grid()
plt.xlabel('Epochs')
plt.title('Model Accuracy')
plt.show()
# Define test_labels before using it
test_labels = test_dataset.class_names

for image, label in normalized_testing_data.take(1):
 model_prediction = model.predict(image)
 for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(image[i].numpy().astype("float")) # dtype should be float after normalization!
    plt.title(f"Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}"+
              f"\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
    plt.grid(True)
    plt.axis("on")
 plt.show()
model.evaluate(normalized_testing_data, verbose=1)
test_labels = test_dataset.class_names
plt.figure(figsize=(25, 20))
for image, label in normalized_testing_data.take(1):
 model_prediction = model.predict(image)
 for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(image[i].numpy().astype("float"))
# Load the new model before using it
try:
    new_model = tf.keras.models.load_model('../input/100-bird-species/EfficientNetB3-birds-98.92.h5')
    
    for image, label in normalized_testing_data.take(1):
        new_model_prediction = new_model.predict(image)
        for i in range(6):
            plt.subplot(1, 6, i+1)
            plt.imshow(image[i].numpy().astype("float")) # dtype should be float after normalization!
            plt.title(f"Prediction: {test_labels[tf.argmax(tf.round(new_model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
            plt.grid(True)
            plt.axis("on")
        plt.show()
    new_model.fit(normalized_training_data, epochs=5, validation_data=normalized_validation_data)
    new_model_evaluation = new_model.evaluate(normalized_testing_data, verbose=1)
except Exception as e:
    print(f"Could not load new model: {e}")
    new_model = None
            plt.axis("on")
        plt.show()
    new_model.fit(normalized_training_data, epochs=5, validation_data=normalized_validation_data)
    new_model_evaluation = new_model.evaluate(normalized_testing_data, verbose=1)
if new_model is not None:
    for image, label in normalized_testing_data.take(1):
        new_model_prediction = new_model.predict(image)
        for i in range(6):
            plt.subplot(1, 6, i+1)
            plt.imshow(image[i].numpy().astype("float")) # dtype should be float after normalization!
if new_model is not None:
    for image, label in normalized_testing_data.take(1):
        new_model_prediction = new_model.predict(image)
        for i in range(6):
            plt.subplot(1, 6, i+1)
            plt.imshow(image[i].numpy().astype("float")) # dtype should be float after normalization!
            plt.title(f"Prediction: {test_labels[tf.argmax(tf.round(new_model_prediction[i]))]}" + 
                    f"\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
            plt.axis("on")
        plt.show()
else:
    try:
        new_model = tf.keras.models.load_model('../input/100-bird-species/EfficientNetB3-birds-98.92.h5')
        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    except Exception as e:
        print(f"Could not load new model: {e}")
        new_model = None
              f"\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
    plt.axis("on")
 plt.show()
new_model = tf.keras.models.load_model('../input/100-bird-species/EfficientNetB3-birds-98.92.h5')





new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
new_model.fit(normalized_training_data, epochs=5, validation_data=normalized_validation_data)
new_model_evaluation = new_model.evaluate(normalized_testing_data, verbose=1)
test_labels = test_dataset.class_names
plt.figure(figsize=(25, 20))
for image, label in normalized_testing_data.take(1):
 new_model_prediction = new_model.predict(image)
 for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(image[i].numpy().astype("float")) # dtype should be float after normalization!
    plt.title(f"Prediction: {test_labels[tf.argmax(tf.round(new_model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
    plt.grid(True)
    plt.axis("on")
 plt.show()