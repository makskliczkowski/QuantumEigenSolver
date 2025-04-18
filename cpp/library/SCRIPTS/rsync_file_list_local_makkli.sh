#!bin/bash
file=$1
target_folder="/home/makkli4548/mylustre/TMP/"
while IFS= read -r dir; do
    rsync -av --progress "$dir"/* "$target_folder"
done < ${file}
