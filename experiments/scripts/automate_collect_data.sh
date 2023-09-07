#!/bin/bash

# Loop from 12 to 19
for i in {12,15,16}
do
    python /home/roblab20/dev/allsight/allsight_sim/experiments/01_collect_data_sim.py summary.sensor_id=$i
done

# Check if the "sensors_idx" argument is provided
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 --sensors_idx <index1> <index2> <index3> ..."
#     exit 1
# fi

# # Search for "--sensors_idx" argument
# if [[ "$0" == "--sensors_idx" ]]; then
#     shift  # Remove the "--sensors_idx" argument from the list
#     # Loop through provided indexes
#     for i in "$@"
#     do
#         python /home/roblab20/dev/allsight/allsight_sim/experiments/01_collect_data_sim.py summary.sensor_id=$i
#     done
# else
#     echo "Usage: $0 --sensors_idx <index1> <index2> <index3> ..."
#     exit 1
# fi
