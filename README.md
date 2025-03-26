
# Nebius Fine Tuning

Prepare an end-to-end example of multi-node fine-tuning of
an LLM




## K8s Cluster Setup

- 2 H100 GPU cards
- 32 vCPU
- 400GB RAM
- 2TB SSD network disk
- 2TB SSD shared filesystem

## Create Control Plane
```bash
  nebius mk8s cluster create \
  --name cluster-example \
  --control-plane-version 1.30 \
  --control-plane-subnet-id \
    $(nebius vpc subnet list --format json \
    | jq -r '.items[0].metadata.id') \
  --control-plane-endpoints-public-endpoint=true
```
