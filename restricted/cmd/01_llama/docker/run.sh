#!/bin/bash

set -e

if [ -d "data" ]; then
    echo "Data directory already exists. Launching main program..."
    ./llama3_1b
    exit 0
fi



command -v wget >/dev/null 2>&1 || { echo >&2 "Error: Please install wget first"; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo >&2 "Error: Please install unzip first"; exit 1; }


rm -rf llama3_1b_data.zip.*  llama3_1b_data.zip
echo "Downloading data segments..."
wget -c https://github.com/Jimmy2099/torch/releases/download/llama3_1b_data/llama3_1b_data.zip.001
wget -c https://github.com/Jimmy2099/torch/releases/download/llama3_1b_data/llama3_1b_data.zip.002


echo "Merging files..."
cat llama3_1b_data.zip.* > llama3_1b_data.zip

echo "Extracting archive..."
unzip -o llama3_1b_data.zip


echo "Cleaning temporary files..."
rm -rf llama3_1b_data.zip.*  llama3_1b_data.zip

echo "Initialization complete. Starting main program..."
./llama3_1b
