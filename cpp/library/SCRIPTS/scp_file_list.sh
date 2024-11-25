#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <remote> <file_with_directories>"
  echo "Example: $0 user@remote:/path/to/destination directories.txt"
  exit 1
fi

# Input arguments
REMOTE=$1
DIRECTORIES_FILE=$2

# Check if the file exists
if [ ! -f "$DIRECTORIES_FILE" ]; then
  echo "Error: File '$DIRECTORIES_FILE' not found."
  exit 1
fi

# Loop through the list of directories and copy them
while IFS= read -r DIR; do
  if [ -d "$DIR" ]; then
    echo "Copying directory: $DIR to $REMOTE"
    scp -r --remove-source-files "$DIR" "$REMOTE"
  else
    echo "Warning: Directory '$DIR' does not exist. Skipping."
  fi
done < "$DIRECTORIES_FILE"

echo "Batch SCP completed."
