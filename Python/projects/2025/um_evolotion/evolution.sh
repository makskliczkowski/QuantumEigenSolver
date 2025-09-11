#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
echo "Running evolution script in $(pwd)"
chmod +x ./evolution.py

# run with:
# 1. save dir
# 2. alpha_start
# 3. alpha_step
# 4. alpha_num
# 5. number of realizations
# 6. Ns_start
# 7. Ns_stop
# 8. additional parameter
# 9. number of timesteps
# 10. memory per job
# 11. maximal memory
    # --save_dir data_big_times_fixed_f \
python ./evolution.py \
    --save_dir data_log \
    --alpha_start 1.5 \
    --alpha_step 0.04 \
    --alphas_number 1 \
    --sites_start 11 \
    --sites_end 11 \
    --number_of_realizations 5 \
    --n 1 \
    --time_num 300000 \
    --max_memory 64 \
    --memory_per_worker 16 \
    --model rpm \
    --uniform 1
