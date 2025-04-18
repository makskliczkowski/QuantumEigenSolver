#!bin/bash
file=$1
target_folder="/home/klimak97/mylustre-hpc-maciek/TMP/"
while IFS= read -r dir; do
    rsync -av --progress "$dir"/* "$target_folder"
done < ${file}
