#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
echo "Running evolution script in $(pwd)"
echo "Files in this directory:"
ls -l

chmod +x ./evolution.py

# run with:
# 1. save dir
# 2. alpha_start
# 3. alpha_step
# 4. alpha_num
# 5. number of realizations
# 6. Ns_start
# 7. Ns_step
# 8. additional parameter
# 9. number of timesteps
# 10. memory per job
# 11. maximal memory
python ./evolution.py      \
    data_big_times         \
    0.7                    \
    0.05                   \
    10                     \
    1                     \
    7                      \
    8                      \
    1                      \
    100000                 \
    64                     \
    64                     \
    -m plrb
