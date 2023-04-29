'''
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Fine-Tune the SwinV2 (tiny) model using the THFOOD-50 dataset.
'''
from PIL import ImageFile
from datasets import load_dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import sys
import torch

def data_preprocessors(image_processor):
  '''
  Define Image Transformations to train vs. test/val datasets,
  and return correspondings data preprocessing functions
  '''
  normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
  size = (
      image_processor.size["shortest_edge"]
      if "shortest_edge" in image_processor.size
      else (image_processor.size["height"], image_processor.size["width"])
  )
  train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
  )
  val_transforms = Compose(
          [
              Resize(size),
              CenterCrop(size),
              ToTensor(),
              normalize,
          ]
  )

  def preprocess_train(example_batch):
      example_batch["pixel_values"] = [
          train_transforms(image.convert("RGB")) for image in example_batch["image"]
      ]
      return example_batch

  def preprocess_val(example_batch):
      example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
      return example_batch
  return preprocess_train, preprocess_val

def collate_fn(batch_examples):
    '''
    Collates batches of examples
    '''
    pixel_values = torch.stack([example["pixel_values"] for example in batch_examples])
    labels = torch.tensor([example["label"] for example in batch_examples])
    return {"pixel_values": pixel_values, "labels": labels}

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    '''
    Compute Accuracy using model predictions
    '''
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# CONSTANTS
CHECKPOINT = "microsoft/swinv2-tiny-patch4-window8-256"
BATCH_SIZE = 64

def main(argv):
  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  NUM_EPOCHS = int(argv[1])

  # Load Dataset
  dataset = load_dataset("thean/THFOOD-50")
  
  # Extract Label and Id mappings
  labels = dataset["train"].features["label"].names
  label2id, id2label = dict(), dict()
  for i, label in enumerate(labels):
      label2id[label] = i
      id2label[i] = label

  # Preprocess and Transform Data (Data Augmentations)
  image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
  preprocess_train, preprocess_val = data_preprocessors(image_processor)
  train_ds = dataset["train"]
  val_ds = dataset["val"]
  test_ds = dataset["test"]

  train_ds.set_transform(preprocess_train)
  val_ds.set_transform(preprocess_val)
  test_ds.set_transform(preprocess_val)

  # Load Model
  model = AutoModelForImageClassification.from_pretrained(
      CHECKPOINT,
      num_labels=len(labels),
      id2label=id2label,
      label2id=label2id,
      ignore_mismatched_sizes = True,
    ).to(DEVICE)
  
  # Define Training Args and Trainer
  if len(argv)>=3:
    output_dir = argv[2]
  else:
    model_name = CHECKPOINT.split("/")[-1]
    output_dir = f"{model_name}-finetuned-THFOOD-50"

  training_args = TrainingArguments(
      output_dir = output_dir,
      remove_unused_columns=False,
      evaluation_strategy = "epoch",
      save_strategy = "epoch",
      learning_rate=5e-5,
      per_device_train_batch_size=BATCH_SIZE,
      gradient_accumulation_steps=4,
      per_device_eval_batch_size=BATCH_SIZE,
      num_train_epochs=NUM_EPOCHS,
      warmup_ratio=0.1,
      logging_steps=10,
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
  )
  trainer = Trainer(
      model,
      training_args,
      data_collator=collate_fn,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      tokenizer=image_processor,
      compute_metrics=compute_metrics,
  )

  # Train Model
  ImageFile.LOAD_TRUNCATED_IMAGES = True  # To allow loading large files
  trainer.train()

  # Evaluate Train, Val and Test data
  metrics = trainer.evaluate(train_ds)
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)

  metrics = trainer.evaluate(val_ds)
  trainer.log_metrics("val", metrics)
  trainer.save_metrics("val", metrics)

  metrics = trainer.evaluate(test_ds)
  trainer.log_metrics("test", metrics)
  trainer.save_metrics("test", metrics)
  
  # Save Model
  trainer.save_state()
  trainer.save_model()

  return 

if __name__ == "__main__":
  if len(sys.argv)<2:
    print("Usage: python swinv2.py 20 /swinv2-finetuned-THFOOD-50")
    print("The first argument is the Number of Epochs")
    print("The second argument is the Output Directory. This is optional.")
  else:
    main(sys.argv)
