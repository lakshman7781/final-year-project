#!/bin/bash

# Exit on error
set -e

echo "Starting AWS Setup..."

# 1. Update System
echo "Updating system packages..."
sudo yum update -y || sudo apt-get update -y

# 2. Install Python 3 and Pip
echo "Installing Python 3..."
sudo yum install python3 python3-pip -y || sudo apt-get install python3 python3-pip -y

# 3. Create Virtual Environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv
source venv/bin/activate

# 4. Install Dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install kaggle

# 5. Local Data Setup
echo "----------------------------------------------------------------"
echo "Setting up Data..."
echo "----------------------------------------------------------------"

mkdir -p data

if [ -f data/True.csv.zip ] && [ -f data/Fake.csv.zip ]; then
    echo "Found local zip files. Extracting..."
    sudo yum install unzip -y || sudo apt-get install unzip -y
    unzip -o data/True.csv.zip -d data/
    unzip -o data/Fake.csv.zip -d data/
    echo "Extraction complete."
else
    # Fallback to Kaggle download if zips are missing
    echo "Zip files not found. Attempting Kaggle download..."
    
    mkdir -p ~/.kaggle

    if [ ! -f ~/.kaggle/kaggle.json ]; then
        read -p "Enter Kaggle Username: " KAGGLE_USERNAME
        read -p "Enter Kaggle Key: " KAGGLE_KEY
        
        echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
    fi

    echo "Downloading Fake and Real News Dataset..."
    kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset --unzip -p data/
fi

echo "Setup Complete! To activate environment run: source venv/bin/activate"
