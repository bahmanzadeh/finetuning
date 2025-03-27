
# Nebius Fine Tuning

Prepare an end-to-end example of multi-node fine-tuning of
an LLM

## K8s Cluster Setup
### Create a two nodes K8s cluster with total capacity as below:

- 2 H100 GPU cards
- 32 vCPU
- 400GB RAM
- 2TB SSD network disk
- 2TB SSD shared filesystem

## Create Control Plane
```bash
  nebius mk8s cluster create \
  --name reza-k8s1 \
  --control-plane-version 1.30 \
  --control-plane-subnet-id \
    $(nebius vpc subnet list --format json \
    | jq -r '.items[0].metadata.id') \
  --control-plane-endpoints-public-endpoint=true
```
## Get the cluster id
```bash
    export NB_K8S_CLUSTER_ID=$(nebius mk8s cluster get-by-name \
    --name reza-k8s1 --format json | jq -r '.metadata.id')
    echo $NB_K8S_CLUSTER_ID
```

## Create Node Group
```bash
    nebius mk8s node-group create \
    --parent-id $NB_K8S_CLUSTER_ID \
    --name mk8s-node-group-reza \
    --fixed-node-count 2 \
    --template-resources-platform gpu-h100-sxm \
    --template-resources-preset 1gpu-16vcpu-200gb \
    --template-gpu-settings-drivers-preset cuda12
```