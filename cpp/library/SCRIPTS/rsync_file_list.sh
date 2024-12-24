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

# Cleanup function for Ctrl+C
cleanup() {
  echo "Script terminated by user (Ctrl+C). Cleaning up..."
  exit 130
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Function to perform rsync
perform_rsync() {
  local SRC=$1
  local DST=$2

  if [ -n "$PASSWORD" ]; then
    if $REMOVE_SOURCE_FILES; then
      sshpass -p "$PASSWORD" rsync -rv --ignore-existing --remove-source-files -e 'ssh -p 22' "$SRC" "$DST"
      # find "$SRC" -type d -empty -delete
    else
      sshpass -p "$PASSWORD" rsync -rv --ignore-existing -e 'ssh -p 22' "$SRC" "$DST"
    fi
  else
    if $REMOVE_SOURCE_FILES; then
      rsync -rv --ignore-existing --remove-source-files -e 'ssh -p 22' "$SRC" "$DST"
      # find "$SRC" -type d -empty -delete
    else
      rsync -rv --ignore-existing -e 'ssh -p 22' "$SRC" "$DST"
    fi
  fi
}

# Loop through the list of directories and copy them
while IFS= read -r REMOTE_DIR; do
  # Construct the full remote directory path
  REMOTE_DEST="$REMOTE/$REMOTE_DIR"

  echo "Syncing from remote directory: $REMOTE_DEST to local"
  
  # Perform the rsync from remote to local
  perform_rsync "$REMOTE_DEST/" "$LOCAL_DIRECTORY/"
  
  if [ $? -eq 0 ]; then
    echo "Successfully synced from: $REMOTE_DEST"
  else
    echo "Error syncing from: $REMOTE_DEST"
  fi
done < "$DIRECTORIES_FILE"

echo "Batch rsync completed."
