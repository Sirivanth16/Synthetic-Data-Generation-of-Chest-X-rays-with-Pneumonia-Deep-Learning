# train.py

import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import data_processing as dp  # Assuming data_processing is in the same directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import boto3
import argparse
import tarfile
from model_definition import create_model


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
S3_BUCKET = 'cv-project-bucket-1'
MODEL_DIRECTORY = '/opt/ml/model/'
MODEL_H5_NAME = 'best_model.h5'
MODEL_TAR_NAME = 'best_model.tar.gz'
MODEL_SAVE_PATH = os.path.join(MODEL_DIRECTORY, MODEL_H5_NAME)
MODEL_TAR_PATH = os.path.join(MODEL_DIRECTORY, MODEL_TAR_NAME)
MODEL_S3_KEY_PREFIX = 'model/best_model'
MODEL_S3_H5_KEY = f"{MODEL_S3_KEY_PREFIX}.h5"
MODEL_S3_TAR_KEY = f"{MODEL_S3_KEY_PREFIX}.tar.gz"

S3_PROCESSED_DATA_PATH = 'processed_data/chest_xray/'
TRAIN_DATA_PATH = os.path.join(dp.EXTRACT_FOLDER, 'train/')
VAL_DATA_PATH = os.path.join(dp.EXTRACT_FOLDER, 'val/')


def load_data():
    try:
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(TRAIN_DATA_PATH,
                                                      target_size=(150, 150),
                                                      batch_size=32,
                                                      class_mode='binary')
        val_gen = val_datagen.flow_from_directory(VAL_DATA_PATH,
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='binary')
        return train_gen, val_gen
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None, None


# Argument parsing for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)  # Default value is arbitrary
args, _ = parser.parse_known_args()


def save_and_compress_model(model, model_save_path, compressed_model_path):
    """
    Save the model to the specified path in .h5 format and then compresses it into .tar.gz.
    """
    # Ensure the model directory exists
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    
    model.save(model_save_path)
    with tarfile.open(compressed_model_path, "w:gz") as tar:
        tar.add(model_save_path, arcname=os.path.basename(model_save_path))

def train_model(train_gen, val_gen):
    try:
        model = create_model()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stopping])

        for epoch, loss in enumerate(history.history['loss']):
            print(f"Epoch {epoch + 1}: Training loss: {loss:.4f}")

        # Save and compress the trained model
        save_and_compress_model(model, MODEL_SAVE_PATH, MODEL_TAR_PATH)
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")


def upload_to_s3(local_path, s3_bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, s3_bucket, s3_key)


def download_from_s3(s3_directory, local_directory):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(S3_BUCKET)
    
    # Ensure local directory exists
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
        

    # List and download each file
    for obj in bucket.objects.filter(Prefix=s3_directory):

        # Construct the local file path based on the S3 object key
        s3_relative_path = os.path.relpath(obj.key, S3_PROCESSED_DATA_PATH)
        local_file_path = os.path.join(local_directory, s3_relative_path)
        
        # Ensure the specific class directory (NORMAL or PNEUMONIA) exists
        local_class_directory = os.path.dirname(local_file_path)
        if not os.path.exists(local_class_directory):
            os.makedirs(local_class_directory)

        bucket.download_file(obj.key, local_file_path)


def main():
    logger.info(f"Downloading train data from {TRAIN_DATA_PATH}...")
    download_from_s3(os.path.join(S3_PROCESSED_DATA_PATH, 'train/'), TRAIN_DATA_PATH)
    logger.info(f"Downloaded train data to {TRAIN_DATA_PATH}")

    logger.info(f"Downloading validation data from {VAL_DATA_PATH}...")
    download_from_s3(os.path.join(S3_PROCESSED_DATA_PATH, 'val/'), VAL_DATA_PATH)
    logger.info(f"Downloaded validation data to {VAL_DATA_PATH}")

    train_gen, val_gen = load_data()
    if train_gen and val_gen:
        train_model(train_gen, val_gen)
        upload_to_s3(MODEL_TAR_PATH, S3_BUCKET, MODEL_S3_TAR_KEY)

if __name__ == "__main__":
    main()