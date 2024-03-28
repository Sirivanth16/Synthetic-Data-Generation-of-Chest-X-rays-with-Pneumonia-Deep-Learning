# data_processing.py

import os
import boto3
import logging
import argparse
import zipfile
from botocore.exceptions import ClientError

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 paths and local paths
S3_BUCKET = 'cv-project-bucket-1'
S3_PROCESSED_DATA_PATH = 'processed_data/'
TEMP_ZIP_PATH = '/opt/ml/processing/input1/archive.zip'
EXTRACT_FOLDER = '/opt/ml/processing/input1/chest_xray/'


# Argument parsing for SageMaker paths
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, default=os.environ.get('SM_CHANNEL_INPUT_DATA', '/tmp/'))
parser.add_argument('--output-data', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/tmp/'))
args, _ = parser.parse_known_args()

EXTRACT_FOLDER = os.path.join(args.output_data, 'chest_xray/')


# Initialize S3 client
s3 = boto3.client('s3')

def unzip_file(zip_path, extract_to):
    """Unzip an archive to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def upload_to_s3(local_folder, s3_path):
    """Recursively upload a local folder to S3."""
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(S3_BUCKET)

    for subdir, _, files in os.walk(local_folder):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                key = os.path.join(s3_path, full_path[len(local_folder):])
                bucket.put_object(Key=key, Body=data)

def main():
    try:
        # Download the zip file from S3 to the local temp path
        logger.info("Starting the download of archive.zip from S3...")
        s3.download_file(S3_BUCKET, 'archive.zip', TEMP_ZIP_PATH)
        logger.info(f"Downloaded archive.zip to {TEMP_ZIP_PATH}")
        
        # Extract the zip file's contents
        logger.info("Extracting archive.zip...")
        unzip_file(TEMP_ZIP_PATH, EXTRACT_FOLDER)
        logger.info(f"Data extracted successfully to {EXTRACT_FOLDER}")

        # remove the temp zip file
        os.remove(TEMP_ZIP_PATH)
        logger.info(f"Removed temporary file: {TEMP_ZIP_PATH}")
        
        logger.info("Uploading processed data to S3...")
        upload_to_s3(EXTRACT_FOLDER, S3_PROCESSED_DATA_PATH)
        logger.info(f"Processed data uploaded to {S3_BUCKET}/{S3_PROCESSED_DATA_PATH}")

    except Exception as e:
        logger.error(f"Error during data extraction from archive.zip: {e}")

if __name__ == "__main__":
    main()