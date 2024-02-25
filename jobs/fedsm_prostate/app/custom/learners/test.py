import pdb, os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts


datalist_path = "D:\\Codes\\ML\\fed-sm\\data\\data_list"
dataset_path = "D:\\Codes\\ML\\fed-sm\\data\\data_set"
# Get datalist json


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
)
traindata = ImageTextContrastiveDataset(datalist_path= datalist_path, dataset_path= dataset_path,imgtransform=transform)
train_collate_fn = ImageTextContrastiveCollator()
trainloader = DataLoader(traindata,
                         batch_size=10,
                         collate_fn=train_collate_fn,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=1,
                         )
cls_prompts = generate_chexpert_class_prompts(n=10)
val_data = ZeroShotImageDataset(['chexpert_5x200'],
    class_names=constants.CHEXPERT_COMPETITION_TASKS)
val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
    mode='multiclass')
eval_dataloader = DataLoader(val_data,
    batch_size= 20,
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    )
print("done")