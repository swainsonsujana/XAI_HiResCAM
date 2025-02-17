from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import nibabel as nib
from scipy import ndimage
import os
import zipfile
import numpy as np
import keras
from keras import layers
from keras import regularizers
from scipy import ndimage
from tensorflow.python.keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def normalize_map(volume):
    """Normalize the volume"""
    min = 0
    max = 1.5
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    #volume = (volume/255.0)*1.5
    return volume
def resize_map(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 60
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

#fetch autistic images from database
autisticpaths= [
    os.path.join(os.getcwd(), "./resampled/resampled-aut", x)
    for x in os.listdir("./resampled/resampled-aut")
    ]
print("Number of scans: " + str(len(autisticpaths)))

#fetch control images from database
nonautisticpaths = [
    os.path.join(os.getcwd(), "./resampled/resampled-nonaut", x)
    for x in os.listdir("./resampled-nonaut")
]
print("Number of scans: " + str(len(nonautisticpaths)))

#Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
autistic_scans = np.array([process_scan(path) for path in autisticpaths])
nonautistic_scans = np.array([process_scan(path) for path in nonautisticpaths])

autistic_labels = np.array([1 for _ in range(len(autistic_scans))])
nonautistic_labels = np.array([0 for _ in range(len(nonautistic_scans))])

scans=np.concatenate((autistic_scans,nonautistic_scans),0)
labels=np.concatenate((autistic_labels,nonautistic_labels),0)

# Split data in the ratio 73-27 for training and validation.
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(scans, labels, test_size=0.27, random_state=1, shuffle=True)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

import random
from scipy import ndimage
import torch.functional as F

def train_preprocessing(volume, label):
  volume = tf.expand_dims(volume, axis=3)
  return volume, label

def validation_preprocessing(volume, label):
    #Process validation data by only adding a channel.
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
batch_size = 4
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

def get_model(width=128, height=128, depth=60):
    """Build a modified 3D Lenet"""
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=6, kernel_size=5, activation="relu", strides=(1,1,1), padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(inputs)
    #x = layers.BatchNormalization()(x)
    x = layers.AveragePooling3D(pool_size=2,strides=(2,2,2))(x)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=6, kernel_size=5, activation="relu", strides=(1,1,1), padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.AveragePooling3D(pool_size=2,strides=(2,2,2))(x)

    x = layers.Conv3D(filters=6, kernel_size=5, activation="relu", strides=(1,1,1), padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.AveragePooling3D(pool_size=2,strides=(2,2,2))(x)
    #x = layers.Dropout(0.3)(x)

    x = layers.Dense(units=120, activation="relu", kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.Flatten()(x)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=84, activation="relu", kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    #x = layers.Dropout(0.5)(x)
    outputs= layers.Dense(units=1,kernel_initializer='he_normal', activation="sigmoid")(x)

    model=keras.Model(inputs, outputs, name="LeNet_layer6")
    return model

# Build model.
model = get_model(width=128, height=128, depth=60)
model.summary()

# Compile model.
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["acc"],
    run_eagerly=True
)

# Define callbacks.
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
csv_logger = CSVLogger("./LeNet_layer6_v1.csv", append=True)

#Train the model, doing validation at the end of each epoch
epochs = 100
history=model.fit(
    train_dataset,
    validation_data=validation_dataset,
   epochs=epochs,
    shuffle=True,
    verbose=1,
   callbacks=[csv_logger, early_stopping_cb]
)
model.save("./LeNet_layer6_v1.keras")

#visualizing model's performance
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()


for i, metric in enumerate(["acc","loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

# Evaluate the model
loss, acc =model.evaluate(validation_dataset,verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

from sklearn.metrics import precision_recall_curve, auc
from sklearn.datasets import make_classification
tf.keras.backend.clear_session()
# Step 4: Predict probabilities for the test dataset.
y_scores = model.predict(x_val, batch_size=4)

import sklearn.metrics as metrics
precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
# Step 6: Calculate Area Under the PR curve.
fpr,tpr,_=metrics.roc_curve(y_val,y_scores)
pr_auc = auc(recall, precision)
# Print the PR AUC
print(f'PR AUC: {pr_auc}')
#print('Precision', precision)
#print('Recall', recall)
plt.plot(fpr,tpr,label="AUC="+str(pr_auc))
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc=4)
plt.show()

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
y_pred = np.where(y_scores> 0.5, 1, 0)
#print(y_pred)
print('f1 score :%.2f'% f1_score(y_val, y_pred, average="macro"))
print('Precison :%.2f'%precision_score(y_val, y_pred, average="macro"))
print('Recall :%.2f'%recall_score(y_val, y_pred, average="macro"))
print('Kappa',cohen_kappa_score(y_val,y_pred))

from sklearn.metrics import classification_report
from sklearn.metrics import __all__
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_val,y_pred)
#specificity = specificity_score(matrix)
#sensitive = sensitive_score(matrix)
print(matrix)
print(classification_report(y_val, y_pred))
