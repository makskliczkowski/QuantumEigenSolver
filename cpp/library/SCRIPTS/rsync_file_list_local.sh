#!bin/bash

target_folder="

while IFS= read -r dir; do
    mv "$dir"/* "$target_folder"
done < directories.txt
