
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
## Here is the CLI command in case the json file didn't work
```bash
export NB_K8S_NODE_TEMPLATE=$(cat <<EOF
{
  "spec": {
    "template": {
      "filesystems": [
        {
          "attach_mode": "READ_WRITE",
          "mount_tag": "$MOUNT_TAG",
          "existing_filesystem": {
            "id": "$NB_FS_ID"
          }
        }
      ],
      "cloud_init_user_data": $USER_DATA
    }
  }
}
EOF
```
And now you can run the CLI
```bash
nebius mk8s node-group create \
  --parent-id $NB_K8S_CLUSTER_ID \
  --name mk8s-node-group-reza \
  --fixed-node-count 2 \
  --template-service-account-id $NB_MK8S_SA_ID \
  --template-resources-platform gpu-h100-sxm \
  --template-resources-preset 1gpu-16vcpu-200gb \
  --template-gpu-settings-drivers-preset cuda12 \
  --template-boot-disk-type network_ssd \
  --template-boot-disk-size-gibibytes 1000 \
  -- "$NB_K8S_NODE_TEMPLATE"
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
## Build and Deployment
```bash
docker build -t rezabah/distilbert-ft:v1 .
docker push rezabah/distilbert-ft:v1
kubectl apply -f pvc.yaml -n finetune
kubectl apply -f master-service.yaml -n finetune
kubectl apply -f fine-tune-job.yaml -n finetune
kubectl apply -f tensorboard-deploy.yaml -n finetune
kubectl apply -f tensorboard-service.yaml -n finetune
kubectl port-forward service/tensorboard-service 6006:6006 -n finetune
Tensorboard: http://127.0.0.1:6006/
```
## Monitoring GPU utilization on the pods
```bash
kubectl get pods -n finetune -o name | grep 'fine-tune-job' | xargs -I POD kubectl exec -n finetune POD -- nvidia-smi
```

## Delete the node group
```bash
export NB_K8S_NODE_GROUP_ID=$(nebius mk8s node-group get-by-name \
--parent-id $NB_K8S_CLUSTER_ID \
--name mk8s-node-group-reza --format json | jq -r '.metadata.id')
echo $NB_K8S_NODE_GROUP_ID
nebius mk8s node-group delete --id $NB_K8S_NODE_GROUP_ID
```