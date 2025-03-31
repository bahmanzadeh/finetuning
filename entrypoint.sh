#!/bin/bash
# Entry point script for distributed training with accelerate

# Extract the NODE_RANK from the pod hostname 
POD_NAME=$(hostname)
export NODE_RANK=$(echo $POD_NAME | awk -F'-' '{print $NF}')

# Run the accelerate launcher
exec accelerate launch \
    --num_processes ${NPROC_PER_NODE} \
    --num_machines ${NNODES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    --dynamo_backend no \
    fine-tune.py "$@"