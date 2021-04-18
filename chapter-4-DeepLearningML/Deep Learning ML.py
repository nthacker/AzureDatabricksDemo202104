# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Healthcare Demo
# MAGIC ## Chapter 4: Deep Learning & Pneumonia Detection

# COMMAND ----------

# MAGIC %md
# MAGIC ![image info](https://visualassets.blob.core.windows.net/flow-diagrams/deep-learning.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC As we discovered at the end of the last chapter, we have data sets available that will help us move from building reactive ML models, like our patient length of stay, to proactive ML models, like helping to accelerate the diagnosis of common ailments with a view to reducing hospital stays. If Contoso Hospital can help their physicians to diagnose pneumonia more quickly, then it can reduce their workload and enable treatment to start more rapidly. This is good for the hospital, the physician, and the patient.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC Let's remind ourselves of the data we have available and prepare our environment for building a PyTorch-based deep learning model for pnemonia diagnosis.

# COMMAND ----------

# DBTITLE 1,List mounted directory
dbutils.fs.ls("/mnt/xray")

# COMMAND ----------

# DBTITLE 1,Import libraries
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
import mlflow
import tempfile
from tensorboardX import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Define functions for inferencing
def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image
  
def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    #img = np.array(img).transpose((2, 0, 1)) / 256
    img = np.array(img) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
  
def predict(image_path, model, topk=2):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)
    
    img_tensor = img_tensor.view(1, 3, 224, 224).cuda(0) #Ping gpu id

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)
        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class
      
def display_prediction(image_path, model, topk):
    """Display image and preditions from model"""

    # Get predictions
    img, ps, classes, y_obs = predict(image_path, model, topk)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes)

    # Show the image
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    ax, img = imshow_tensor(img, ax=ax)

    # Set title to be the actual class
    ax.set_title(y_obs, size=20)

    ax = plt.subplot(1, 2, 2)
    # Plot a bar plot of predictions
    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.tight_layout()

# COMMAND ----------

# DBTITLE 1,Display sample training image
ex_img = Image.open('/dbfs/mnt/xray/test/NORMAL/IM-0001-0001.jpeg')
imshow(ex_img)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scripts
# MAGIC 
# MAGIC The next two cells define our training code. Keep an eye out for calls to take advantage of GPUs and/or Horovod-based scale-out training where available. MLflow is also used to manage the end-to-end experiment run including logging and model management.
# MAGIC 
# MAGIC The model is based on ResNet-50, a common convolutional neural network architecture.

# COMMAND ----------

# DBTITLE 1,Define training helper functions
def train(model, epoch, warmup_epochs, train_sampler, train_loader, verbose, optimizer, log_writer, base_lr, batches_per_allreduce, batch_size):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            #adjust_learning_rate(epoch, batch_idx)
            adjust_learning_rate(epoch, warmup_epochs, batch_idx, train_loader, optimizer, base_lr, batches_per_allreduce)

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                target_batch = target[i:i + batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
    
    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        mlflow.log_metric('train/loss', float(train_loss.avg))
        mlflow.log_metric('train/accuracy', float(train_accuracy.avg))
        
def validate(model, epoch, verbose, log_writer, val_loader):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
        mlflow.log_metric('val/loss', float(val_loss.avg))
        mlflow.log_metric('val/accuracy', float(val_accuracy.avg))


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, warmup_epochs, batch_idx, train_loader, optimizer, base_lr, batches_per_allreduce):
    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * hvd.size() * batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def save_checkpoint(epoch, optimizer, model, checkpoint_format):
    if hvd.rank() == 0:
        filepath = checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

# COMMAND ----------

# DBTITLE 1,Define distributed training script
def train_hvd():  
  import mlflow.pytorch
  import pickle
  import numpy

  batch_size = 32
  val_batch_size = 32
  warmup_epochs = 5
  base_lr = 0.0125
  batches_per_allreduce = 1
  checkpoint_format = './checkpoint-{epoch}.pth.tar'
  cuda = torch.cuda.is_available()
  seed = 42 
  epochs = 10
  log_dir = './logs'
  use_adasum = False
  momentum = 0.9
  wd = 0.00005
  fp16_allreduce = False
  gradient_predivide_factor = 1.0
  allreduce_batch_size = batch_size * batches_per_allreduce
  train_dir = '/dbfs/mnt/xray/train/'
  val_dir = '/dbfs/mnt/xray/val'
  test_dir = '/dbfs/mnt/xray/test'
  log_interval = 100

  hvd.init()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(seed)

  if cuda:
      # Horovod: pin GPU to local rank.
      torch.cuda.set_device(hvd.local_rank())
      torch.cuda.manual_seed(seed)

  #cudnn.benchmark = True

  # If set > 0, will resume training from a given checkpoint.
  resume_from_epoch = 0
  for try_epoch in range(epochs, 0, -1):
      if os.path.exists(checkpoint_format.format(epoch=try_epoch)):
          resume_from_epoch = try_epoch
          break

  # Horovod: broadcast resume_from_epoch from rank 0 (which will have
  # checkpoints) to other ranks.
  resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                    name='resume_from_epoch').item()

  # Horovod: print logs on the first worker.
  verbose = 1 if hvd.rank() == 0 else 0

  # Horovod: write TensorBoard logs on first worker.
  log_writer = SummaryWriter(log_dir) if hvd.rank() == 0 else None

  # Horovod: limit # of CPU threads to be used per worker.
  torch.set_num_threads(4)

  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
  # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
  # issues with Infiniband implementations that are not fork-safe
  if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
          mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
      kwargs['multiprocessing_context'] = 'forkserver'

  train_dataset = \
      datasets.ImageFolder(train_dir,
                           transform=transforms.Compose([
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                           ]))
  # Horovod: use DistributedSampler to partition data among workers. Manually specify
  # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=allreduce_batch_size,
      sampler=train_sampler, **kwargs)

  val_dataset = \
      datasets.ImageFolder(val_dir,
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                           ]))
  val_sampler = torch.utils.data.distributed.DistributedSampler(
      val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,
                                           sampler=val_sampler, **kwargs)


  # Set up standard ResNet-50 model.
  model = models.resnet50().to(device)

  # Freeze early layers
  #for param in model.parameters():
  #    param.requires_grad = False

  model.class_to_idx = train_dataset.class_to_idx
  model.idx_to_class = {
      idx: class_
      for class_, idx in model.class_to_idx.items()
  }

  # By default, Adasum doesn't need scaling up learning rate.
  # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
  lr_scaler = batches_per_allreduce * hvd.size() if not use_adasum else 1

  if cuda:
      # Move model to GPU.
      model.cuda()
      # If using GPU Adasum allreduce, scale learning rate by local_size.
      if use_adasum and hvd.nccl_built():
          lr_scaler = batches_per_allreduce * hvd.local_size()

  # Horovod: scale learning rate by the number of GPUs.
  optimizer = optim.SGD(model.parameters(),
                        lr=(base_lr *
                            lr_scaler),
                        momentum=momentum, weight_decay=wd)

  # Horovod: (optional) compression algorithm.
  compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

  # Horovod: wrap optimizer with DistributedOptimizer.
  optimizer = hvd.DistributedOptimizer(
      optimizer, named_parameters=model.named_parameters(),
      compression=compression,
      backward_passes_per_step=batches_per_allreduce,
      op=hvd.Adasum if use_adasum else hvd.Average,
      gradient_predivide_factor=gradient_predivide_factor)

  # Restore from a previous checkpoint, if initial_epoch is specified.
  # Horovod: restore on the first worker which will broadcast weights to other workers.
  if resume_from_epoch > 0 and hvd.rank() == 0:
      filepath = checkpoint_format.format(epoch=resume_from_epoch)
      checkpoint = torch.load(filepath)
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])

  # Horovod: broadcast parameters & optimizer state.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)

  mlflow.set_experiment("/Projects/sebastian@zarmada.net/azure-databricks/chapter-4-DeepLearningML/Pneumonia Image Detection")
  with mlflow.start_run() as run: 
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Val Batch Size', val_batch_size)
    mlflow.log_param('Warmup Epochs', warmup_epochs)
    mlflow.log_param('Base LR', base_lr)
    mlflow.log_param('Batches All Reduce', batches_per_allreduce)
    mlflow.log_param('CUDA Available', cuda)
    mlflow.log_param('Seed', seed)
    mlflow.log_param('Epochs', epochs)
    mlflow.log_param('Log Dir', log_dir)
    mlflow.log_param('Use Adasum', use_adasum)
    mlflow.log_param('Momentum', momentum)
    mlflow.log_param('WD', wd)
    mlflow.log_param('FP16 All Reduce', fp16_allreduce)
    mlflow.log_param('Gradient Predivide Factor', gradient_predivide_factor)
    mlflow.log_param('All Reduce Batch Size', allreduce_batch_size)
    mlflow.log_param('Train Directory', train_dir)
    mlflow.log_param('Validation Directory', val_dir)

    for epoch in range(resume_from_epoch, epochs):
        train(model, epoch, warmup_epochs, train_sampler, train_loader, verbose, optimizer, log_writer, base_lr, batches_per_allreduce, batch_size)
        validate(model, epoch, verbose, log_writer, val_loader)
        save_checkpoint(epoch, optimizer, model, checkpoint_format)

    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(log_dir, artifact_path="events")
    print(
        "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
        % os.path.join(mlflow.get_artifact_uri(), "events")
    )

    # Log the model as an artifact of the MLflow run.
    print("\nLogging the trained model as a run artifact...")
    mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)
    print(
        "\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training
# MAGIC 
# MAGIC Let's train our model. As expected, using distributed training results in a faster runtime. Horovod, CUDA, and PyTorch are built-in and preconfigured as part of the Databricks ML image making it easy to create a powerful, scale-out training run. And with Azure's wide variety of cluster types, GPU training can be enabled on-demand.

# COMMAND ----------

# DBTITLE 1,Non-distributed training
train_hvd()

# COMMAND ----------

# DBTITLE 1,Distributed training with Horovod 
from sparkdl import HorovodRunner

hr = HorovodRunner(np=2) 
hr.run(train_hvd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC 
# MAGIC Let's inspect the most recent run. By fetching logging data managed by MLflow as part of the experiment run, we can also visualize the data through Tensorboard (also built-in to the image).

# COMMAND ----------

# DBTITLE 1,Fetch latest runs
from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(3667476102457265)

# COMMAND ----------

# DBTITLE 1,Download MLflow artifacts
local_dir = "/mnt/logs"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
local_path = client.download_artifacts(runs[0].info.run_id, "events", local_dir)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# DBTITLE 1,Tensorboard
from tensorboard import notebook
notebook.start("--logdir {}".format(local_path))

# COMMAND ----------

# DBTITLE 1,Register model
logged_model = runs[0].info.artifact_uri + '/pytorch-model'

# Since the model was logged as an artifact, it can be loaded to make predictions
loaded_model = mlflow.pytorch.load_model(logged_model)

# COMMAND ----------

# DBTITLE 1,Test pneumonia image
display_prediction("/dbfs/mnt/xray/test/PNEUMONIA/person1_virus_6.jpeg", loaded_model, topk=2)

# COMMAND ----------

# DBTITLE 1,Test normal image
display_prediction("/dbfs/mnt/xray/test/NORMAL/IM-0001-0001.jpeg", loaded_model, topk=2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Conclusion
# MAGIC 
# MAGIC With the power of Azure compute and the feature-packed capabilities in the Databricks ML, we can rapidly create a sophisticated image classification model in minutes by leveraging industry-standard deep learning frameworks, MLflow for end-to-end model management, GPUs for optimized execution, and Horovod scale out training.
# MAGIC 
# MAGIC For Contoso Hospital, this means they can start to tackle non-trivial machine learning problems, like image classification, and enable their physicians to accelerate treatment for their patients, further optimizing the use of critical resources.
