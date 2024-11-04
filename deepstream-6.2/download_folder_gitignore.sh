#!/bin/bash

# Set the version and file details
VER_DS=6.2
FILE_ID="1bwpNw7ZzN4hTSP1LtNnF3gbLl4t-xfGj"
FILE_NAME="ds${VER_DS}_ignore.zip"

# Function to install gdown if not already installed
install_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install gdown
    else
        echo "gdown is already installed."
    fi
}

# Install gdown
install_gdown

# Download the file using gdown
gdown --id ${FILE_ID} -O ${FILE_NAME}

# Unzip the downloaded file
unzip -o ${FILE_NAME}

# Remove the zip file after extraction
rm -rf ${FILE_NAME}