#!/bin/bash

# Bash script to automate data downloading and analysis

# Define directory for datasets within the home directory
DATASET_DIR="$HOME/Documents/vanna/vanna-cexdata/datasets"

# Check if the directory exists, create it if it doesn't
if [ ! -d "$DATASET_DIR" ]; then
    if ! mkdir -p $DATASET_DIR; then
        echo "Failed to create directory $DATASET_DIR. Check permissions or path."
        exit 1
    fi
fi

# Move to the directory
cd $DATASET_DIR

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 {from_date} {to_date}"
    exit 2
fi

# Variables for date range
FROM_DATE=$1
TO_DATE=$2

cd ../

# Execute the Python script with the provided date range
python3 download_scripts/download_coinbase.py --from_date $FROM_DATE --to_date $TO_DATE

# Check for errors in download script
if [ $? -ne 0 ]; then
    echo "Data download failed. Please check the errors."
    exit 3
fi

echo "Data download completed successfully."

# Unzip .gz files
find $DATASET_DIR -name '*.gz' -exec gunzip {} \;

echo "All .gz files have been unzipped successfully."
