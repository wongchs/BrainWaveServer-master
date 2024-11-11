import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import os
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import random
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, 
    GlobalAveragePooling2D, Multiply, Reshape, LSTM, BatchNormalization
)

def print_files():
    target_path = './kaggle/input'
    if os.path.exists(target_path):
        for dirname, _, filenames in os.walk(target_path):
            for filename in filenames:
                print(os.path.join(dirname, filename))
    else:
        print(f"The directory '{target_path}' does not exist.")
        
        
def load_patient_data(patient_num, types, num_segments):
    all_X = []
    all_Y = []
    base_path = './kaggle/input/seizure-prediction/Patient_{}/Patient_{}/'.format(patient_num, patient_num)

    for i, typ in enumerate(types):
        for j in range(num_segments):
            fl = os.path.join(base_path, '{}_{}.mat'.format(typ, str(j + 1).zfill(4)))
            data = scipy.io.loadmat(fl)
            k = typ.replace(f'Patient_{patient_num}_', '') + '_'
            d_array = data[k + str(j + 1)][0][0][0]
            
            lst = list(range(3000000))  # Adjust for 10 minutes
            for m in lst[::5000]:  # Create a spectrogram every 1 second (5000 samples)
                p_secs = d_array[0][m:m+5000]
                p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
                p_SS = np.log1p(p_Sxx)
                arr = p_SS[:] / np.max(p_SS)
                all_X.append(arr)
                all_Y.append(i)
    
    return all_X, all_Y


# Load both Patient 1 and Patient 2 data
types = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']
all_X1, all_Y1 = load_patient_data(1, types, 18)

types = ['Patient_2_interictal_segment', 'Patient_2_preictal_segment']
all_X2, all_Y2 = load_patient_data(2, types, 18)

# Combine data from both patients
all_X = all_X1 + all_X2
all_Y = all_Y1 + all_Y2

dataset = list(zip(all_X, all_Y))
random.shuffle(dataset)
all_X, all_Y = zip(*dataset)

x_train = np.array(all_X[:42000])
y_train = np.array(all_Y[:42000])
x_test = np.array(all_X[42000:])
y_test = np.array(all_Y[42000:])

img_rows, img_cols = 256, 22
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = 2
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def attention_block(inputs):
    channels = inputs.shape[-1]
    attention = GlobalAveragePooling2D()(inputs)
    attention = Dense(channels // 8, activation='relu')(attention)
    attention = Dense(channels, activation='sigmoid')(attention)
    attention = Reshape((1, 1, channels))(attention)
    attention = Multiply()([inputs, attention])
    return attention

def create_data_pipeline(X, y, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()  # Cache the data in memory
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
    return dataset

def create_model(input_shape=(256, 22, 1), num_classes=2):
    inputs = Input(shape=input_shape)
    
    # CNN layers with batch normalization
    x = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Apply attention mechanism
    x = attention_block(x)
    
    # Flatten and reshape for LSTM
    x = Flatten()(x)
    new_rows = input_shape[0] // 8
    new_cols = input_shape[1] // 8
    features = 128
    x = Reshape((new_rows * new_cols, features))(x)
    
    # LSTM layers
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    
    # Dense layers for classification
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    return Model(inputs, outputs)

train_dataset = create_data_pipeline(x_train, y_train, batch_size=256)  # Increased batch size
test_dataset = create_data_pipeline(x_test, y_test, batch_size=256)

# Create and compile the model
model = create_model()
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

# Print model summary
model.summary()

# Training parameters
batch_size = 128
epochs = 8

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
]

# Train the model
history = model.fit(
    x_train, 
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('seizure_detection_model.keras')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
