#!/bin/bash

# Usage function for help
usage() {
  echo "Usage: $0 [-rs] [-p <password>] <remote> <file_with_directories> <local_directory>"
  echo "Options:"
  echo "  -rs          Remove source files after successful copy"
  echo "  -p <password>  Password for SSH authentication (use with caution)"
  echo "Example: $0 -rs -p yourpassword user@remote:/base/path directories.txt /local/destination"
  exit 1
}

# Default options
REMOVE_SOURCE_FILES=false
PASSWORD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -rs)
      REMOVE_SOURCE_FILES=true
      shift
      ;;
    -p)
      PASSWORD="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# Check remaining arguments
if [ "$#" -lt 3 ]; then
  usage
fi

REMOTE=$1
DIRECTORIES_FILE=$2
LOCAL_DIRECTORY=$3

# Check if the directories file exists
if [ ! -f "$DIRECTORIES_FILE" ]; then
  echo "Error: File '$DIRECTORIES_FILE' not found."
  exit 1
fi

# Check if sshpass is installed
if [ -n "$PASSWORD" ] && ! command -v sshpass &> /dev/null; then
  echo "Error: sshpass is not installed. Install it first."
  exit 1
fi

# Function to perform rsync
perform_rsync() {
  local SRC=$1
  local DST=$2

  if [ -n "$PASSWORD" ]; then
    if $REMOVE_SOURCE_FILES; then
      sshpass -p "$PASSWORD" rsync -rv --ignore-existing --remove-source-files -e 'ssh -p 22' "$SRC" "$DST"
      find "$SRC" -type d -empty -delete
    else
      sshpass -p "$PASSWORD" rsync -rv --ignore-existing -e 'ssh -p 22' "$SRC" "$DST"
    fi
  else
    if $REMOVE_SOURCE_FILES; then
      rsync -rv --ignore-existing --remove-source-files -e 'ssh -p 22' "$SRC" "$DST"
      find "$SRC" -type d -empty -delete
    else
      rsync -rv --ignore-existing -e 'ssh -p 22' "$SRC" "$DST"
    fi
  fi
}

# Loop through the list of directories and copy them
while IFS= read -r DIR; do
  if [ -d "$DIR" ]; then
    # Append the directory path to the remote base path
    REMOTE_DEST="$REMOTE/$DIR"
    echo "Syncing directory: $DIR to $REMOTE_DEST"
    perform_rsync "$DIR/" "$REMOTE_DEST/"
    if [ $? -eq 0 ]; then
      echo "Successfully synced: $DIR"
    else
      echo "Error syncing: $DIR"
    fi
  else
    echo "Warning: Directory '$DIR' does not exist. Skipping."
  fi
done < "$DIRECTORIES_FILE"

# Sync files back from remote to local directory
echo "Syncing back from $REMOTE to $LOCAL_DIRECTORY"
perform_rsync "$REMOTE/" "$LOCAL_DIRECTORY/"

echo "Batch rsync completed."
