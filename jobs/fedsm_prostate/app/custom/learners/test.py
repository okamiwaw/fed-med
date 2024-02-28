import pdb, os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
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
from networks.vgg import vgg11
import torch.optim as optim

datalist_path = "D:\\Codes\\ML\\fed-med\\data\\data_list"
dataset_path = "D:\\Codes\\ML\\fed-med\\data\\data_set"
# Get datalist json

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
)
traindata = ImageTextContrastiveDataset(datalist_path= datalist_path, dataset_path= dataset_path,imgtransform=transform, client_id="client_1")
train_collate_fn = ImageTextContrastiveCollator()
trainloader = DataLoader(traindata,
                         batch_size=20,
                         collate_fn=train_collate_fn,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=0,
                         )

cls_prompts = generate_chexpert_class_prompts(n=10)
val_data = ZeroShotImageDataset(['chexpert_5x200'],
    class_names=constants.CHEXPERT_COMPETITION_TASKS,
    dataset_path= dataset_path)
val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
    mode='multiclass')
eval_dataloader = DataLoader(val_data,
    batch_size= 20,
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    )
device = "cuda:0"
select_model = vgg11(
            num_classes=3,
        ).to(device)
select_criterion = torch.nn.CrossEntropyLoss()
select_optimizer = optim.Adam(
            select_model.parameters(), lr=1e-3
        )
epochs = 10
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(device)
optimizer = optim.Adam(model.parameters(), lr=  2e-5)

## select_model training ##
# def local_train_select(train_loader, select_label, current_round):
#     # Train selector model in full batch manner, and keep track of curves
#     for epoch in range(epochs):
#         select_model.train()
#         epoch_len = len(train_loader)
#         epoch_global = current_round * 1 + epoch
#         progress_bar = tqdm(enumerate(train_loader), total=epoch_len, desc=f"Epoch {current_round}", leave=True)
#         for i, batch_data in progress_bar:
#             inputs = batch_data["pixel_values"].to(device)
#             # construct vector of selector label
#             labels = np.ones(inputs.size()[0], dtype=np.int64) * select_label
#             labels = torch.tensor(labels).to(device)
#             # forward + backward
#             outputs = select_model(inputs)
#             loss = select_criterion(outputs, labels)
#             loss.backward()
#             current_step = epoch_len * epoch_global + i
#             progress_bar.set_postfix({"loss": loss.item()})
#         select_optimizer.step()
#         select_optimizer.zero_grad()
#
# local_train_select(trainloader, 2, 1)

# local training ##
def local_train(
        train_loader,
):
    for epoch in range(epochs):
        loss_model = ImageTextContrastiveLoss(model).to(device)
        loss_model.train()
        epoch_len = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=epoch_len, desc=f"Epoch {epoch} / {epochs}", leave=True)
        for i, batch_data in progress_bar:
            loss_return = loss_model(**batch_data)
            loss = loss_return['loss_value']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})
local_train(trainloader)

#validate process ##
# def local_valid(
#         model,
#         valid_loader,
# ):
#     medclip_clf = PromptClassifier(model)
#     evaluator = Evaluator(
#         medclip_clf=medclip_clf,
#         eval_dataloader=valid_loader,
#         mode='multiclass',
#     )
#     scores = evaluator.evaluate()
#     metric = scores['acc']
#     # tensorboard record id, add to record if provided
#     print(metric)
# local_valid(model, eval_dataloader)

## valid_select process ##
# def AccuracyTopK(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
# def local_valid_select(
#         valid_loader,
#         select_label,
#         current_round=None,
# ):
#     # Validate selector model
#     select_model.eval()
#     with torch.no_grad():
#         metric = 0
#         for i, batch_data in enumerate(valid_loader):
#             # input and expected output
#             images = batch_data["pixel_values"].to(device)
#             # generate label vector: image batch_size, same label
#             select = np.ones(images.size()[0], dtype=np.int64) * select_label
#             select = torch.tensor(select).to(device)
#             # inference
#             outputs = select_model(images)
#             # compute metric
#             metric_score = AccuracyTopK(outputs, select, topk=(1,))
#             metric += metric_score[0].item()
#         # compute mean acc over whole validation set
#         metric /= len(valid_loader)
#     print(metric)
# local_valid_select(eval_dataloader, 2, 1)
