
# Nebius Fine Tuning Project

Prepare an end-to-end example of multi-node fine-tuning of an LLM

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

## Get the service account for the nodegroup
```bash
export NB_MK8S_SA_ID=$(
  nebius iam service-account get-by-name \
    --name k8s-node-group-sa --format json \
    | jq -r '.metadata.id'
)
```

## Create 1TB SSD shared filesystem:

```bash
export NB_FS_ID=$(nebius compute filesystem create \
  --name mk8s-csi-storage \
  --size-gibibytes 2000 \
  --type network_ssd \
  --block-size-bytes 4096 \
  --format json | jq -r ".metadata.id")
```

## Create the cloud-init user data that will mount the shared filesystem to the nodes:
```bash
export MOUNT_POINT=/mnt/data
export MOUNT_TAG=csi-storage
export USER_DATA=$(jq -Rs '.' <<EOF
runcmd:
  - sudo mkdir -p $MOUNT_POINT
  - sudo mount -t virtiofs $MOUNT_TAG $MOUNT_POINT
  - echo $MOUNT_TAG $MOUNT_POINT "virtiofs" "defaults,nofail" "0" "2" | sudo tee -a /etc/fstab
EOF
)
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
-f clusterconfig.json
```

## Get the credential and create kubeconfig
```bash
nebius mk8s cluster get-credentials --id $NB_K8S_CLUSTER_ID --external
```    

## Install the CSI driver
Pull the driver's Helm chart:
```bash
helm pull \
  oci://cr.eu-north1.nebius.cloud/mk8s/helm/csi-mounted-fs-path \
  --version 0.1.2
```
## Install the chart:
```bash
helm upgrade csi-mounted-fs-path ./csi-mounted-fs-path-0.1.2.tgz --install \
  --set dataDir=$MOUNT_POINT/csi-mounted-fs-path-data/
```

## Delete the node group
```bash
export NB_K8S_NODE_GROUP_ID=$(nebius mk8s node-group get-by-name \
--parent-id $NB_K8S_CLUSTER_ID \
--name mk8s-node-group-reza --format json | jq -r '.metadata.id')
echo $NB_K8S_NODE_GROUP_ID
nebius mk8s node-group delete --id $NB_K8S_NODE_GROUP_ID
```