#!/bin/bash

# Usage function for help
usage() {
  echo "Usage: $0 [-rs] <remote> <file_with_directories_on_remote> <local_destination>"
  echo "Options:"
  echo "  -rs    Remove source files on the remote after successful copy"
  echo "Example: $0 -rs user@remote:/base/path directories.txt /local/destination"
  exit 1
}

# Check for required arguments
if [ "$#" -lt 3 ]; then
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
LOCAL_DEST=$3

# Check if the directories file exists
if [ ! -f "$DIRECTORIES_FILE" ]; then
  echo "Error: File '$DIRECTORIES_FILE' not found."
  exit 1
fi

# Check if the local destination directory exists
if [ ! -d "$LOCAL_DEST" ]; then
  echo "Error: Local destination directory '$LOCAL_DEST' does not exist."
  exit 1
fi

# Loop through the list of directories and sync them
while IFS= read -r REMOTE_DIR; do
  FULL_REMOTE_PATH="$REMOTE/$REMOTE_DIR"
  echo "Syncing from remote: $FULL_REMOTE_PATH to local: $LOCAL_DEST"
  
  sudo rsync -rv --rsh --ignore-existing --remove-source-files -e 'ssh -p 22' "$FULL_REMOTE_PATH/" "$LOCAL_DEST/"
  
  if [ $? -eq 0 ]; then
    echo "Successfully synced: $FULL_REMOTE_PATH to $LOCAL_DEST"
  else
    echo "Error syncing: $FULL_REMOTE_PATH"
  fi
done < "$DIRECTORIES_FILE"

echo "Batch rsync from remote completed."
