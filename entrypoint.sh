#!/bin/bash
# Entry point script for distributed training with PyTorch DDP using torchrun

# Debugging information
echo "NODE_RANK=${NODE_RANK}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

if [ "${NODE_RANK}" -eq "0" ]; then
    echo "Labeling the master pod: ${POD_NAME} with role=master"
    kubectl label pod "${POD_NAME}" -n "${NAMESPACE}" role=master --overwrite
fi

exec torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    fine-tune.py "$@"
