
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
kubectl apply -f role.yaml -n fine-tune
kubectl apply -f rolebinding.yaml -n finetune
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
----
# fine-tune.py - Explained
Initialization of Distributed Training (setup function):

This step initializes the process group using torch.distributed.init_process_group. This sets up the communication backend (in this case, "nccl" which is optimized for GPUs) and establishes connections between the different processes (nodes/GPUs) involved in the distributed training. It also sets the current process's GPU device based on its local rank.

Loading Dataset:
The code uses the datasets library to load the "glue" dataset, specifically the "sst2" task (a sentiment analysis dataset). It then shuffles the training data and selects a subset of both the training and validation datasets based on the train_dataset_range and eval_dataset_range arguments.

Loading Tokenizer:
An AutoTokenizer is loaded based on the "distilbert-base-uncased" pretrained model. The tokenizer is responsible for converting the raw text data into numerical inputs that the model can understand.

Loading Model:
An AutoModelForSequenceClassification is loaded, also based on the "distilbert-base-uncased" pretrained model. This model is a DistilBERT model with a classification head on top, suitable for sequence classification tasks like sentiment analysis. The model is then moved to the specific GPU assigned to the current process (local_rank).

Preprocessing Datasets:
A preprocess_function is defined to tokenize the "sentence" column of the datasets. This function truncates and pads the sequences to a maximum length of 128. The map function is then used to apply this preprocessing function to both the training and evaluation datasets in batches.

Synchronization After Preprocessing (dist.barrier()):
This step ensures that all distributed processes have finished the potentially time-consuming data preprocessing step before proceeding further. This is crucial to prevent issues where some processes might be waiting for data that hasn't been fully prepared by others.

Setting Dataset Format:
The format of the training and evaluation datasets is set to "torch", and the relevant columns ("input_ids", "attention_mask", "label") are specified. This prepares the datasets to be used with PyTorch DataLoader.
Creation of train_sampler and eval_sampler:


DistributedSampler is created for both the training and evaluation datasets. These samplers are responsible for partitioning the data across the different processes in the distributed training setup. Each process will only receive a subset of the data based on its rank and the total number of processes (world_size). The train_sampler shuffles the data within its partition for each epoch, while eval_sampler does not.

Creation of train_dataloader and eval_dataloader:
DataLoader instances are created for the training and evaluation datasets. These data loaders provide an iterable over the datasets, yielding batches of data. The sampler argument is set to the respective DistributedSampler, ensuring that each process only iterates over its assigned portion of the data. DataCollatorWithPadding is used to pad the sequences within each batch to a consistent length.

Wrapping the model with DDP:
The loaded model is wrapped with DistributedDataParallel (DDP). This is the core of PyTorch's distributed training capability. DDP handles the communication of gradients between the different processes during the backward pass, ensuring that the model weights are updated consistently across all processes. The device_ids argument specifies which GPU the model should reside on for the current process.

Setting up the optimizer and scheduler:
An AdamW optimizer is initialized with the model's parameters and a specified learning rate. The AdamW optimizer is a common choice for training Transformer models.
A learning rate scheduler (get_scheduler) is set up. In this case, a "linear" scheduler is used, which will linearly decrease the learning rate over the course of training. The num_warmup_steps is set to 0, and num_training_steps is calculated based on the number of epochs and the size of the training data loader.

Entering the training loop:
The code then enters a loop that iterates for the specified number of epochs.

Training Epoch:
Inside each epoch, the model is set to training mode (model.train()).
The train_sampler's epoch is set to the current epoch number to ensure different shuffling of data in each epoch across the distributed processes.
The code iterates through the train_dataloader, processing batches of training data.
For each batch:
The batch is moved to the current GPU.
The model's forward pass is performed to get the output (including the loss).
The gradients are reset using optimizer.zero_grad().
The loss is backpropagated using loss.backward().
The optimizer updates the model's parameters using optimizer.step().
The learning rate scheduler updates the learning rate using lr_scheduler.step().
A progress bar (only visible on rank 0) displays the training progress and the current loss.

Evaluation loop:
After each training epoch, the model is set to evaluation mode (model.eval()).
Variables are initialized to track evaluation loss, correct predictions, and total number of samples.
The code iterates through the eval_dataloader without calculating gradients (with torch.no_grad()).
For each batch:
The batch is moved to the current GPU.
The model's forward pass is performed to get the output logits.
Predictions are obtained by taking the argmax of the logits.
The evaluation loss is calculated using cross-entropy.
The number of correct predictions and the total number of samples are accumulated.

Evaluation Metrics Reporting:
After evaluating the entire evaluation dataset, the average evaluation loss and accuracy are calculated.
If the current process is rank 0, these evaluation metrics are printed.

Saving the Model:
After all epochs are completed, if the current process is rank 0, the trained model's state and the tokenizer's configuration are saved to a specified output directory on the shared filesystem. Saving is typically done only by the master process to avoid redundant saves.

Cleanup (cleanup function):
Finally, the cleanup function is called to destroy the distributed process group, releasing the resources used for distributed training.
