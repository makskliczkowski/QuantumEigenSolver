#!/bin/bash

# Usage function for help
usage() {
  echo "Usage: $0 [-rs] <remote> <file_with_directories>"
  echo "Options:"
  echo "  -rs    Remove source files after successful copy"
  echo "Example: $0 -rs user@remote:/path/to/destination directories.txt"
  exit 1
}

# Check for required arguments
if [ "$#" -lt 2 ]; then
  usage
fi

# Parse arguments
REMOVE_SOURCE_FILES=false
if [ "$1" == "-rs" ]; then
  REMOVE_SOURCE_FILES=true
  shift
fi

REMOTE=$1
DIRECTORIES_FILE=$2

# Check if the directories file exists
if [ ! -f "$DIRECTORIES_FILE" ]; then
  echo "Error: File '$DIRECTORIES_FILE' not found."
  exit 1
fi

# Loop through the list of directories and copy them
while IFS= read -r DIR; do
  if [ -d "$DIR" ]; then
    echo "Syncing directory: $DIR to $REMOTE"
    sudo rsync -rv --rsh --ignore-existing --remove-source-files -e 'ssh -p 22' "$DIR/" "$REMOTE/"
    if [ $? -eq 0 ]; then
      echo "Successfully synced: $DIR"
      if $REMOVE_SOURCE_FILES; then
        echo "Removing empty directories in: $DIR"
        find "$DIR" -type d -empty -delete
      fi
    else
      echo "Error syncing: $DIR"
    fi
  else
    echo "Warning: Directory '$DIR' does not exist. Skipping."
  fi
done < "$DIRECTORIES_FILE"

echo "Batch rsync completed."
