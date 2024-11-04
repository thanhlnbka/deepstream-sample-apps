#!/bin/bash

# Set the version number for the dataset
VER_DS=7.0
# Google Drive file ID for the file to be downloaded
FILE_ID="1yahBUT-UK2kF5HBkCBMKvzeFXk3I6I3d"
# Name of the file to be saved locally
FILE_NAME="ds${VER_DS}_ignore.zip"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown  # Install gdown using pip
fi

# Use gdown to download the file from Google Drive
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O ${FILE_NAME}

# Unzip the downloaded file
unzip -o ${FILE_NAME}

# Remove the zip file after extraction
rm -rf ${FILE_NAME}