# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
import torch.optim as optim
from learners.supervised_learner import SupervisedLearner
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet

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

from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants


class SupervisedMonaiProstateLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """MONAI Learner for prostate segmentation task.
        It inherits from SupervisedLearner.

        Args:
            train_config_filename: path for config_3 file, this is an addition term for config_3 loading
            aggregation_epochs: the number of training epochs for a round.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__(
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
        )
        self.train_config_filename = train_config_filename
        self.config_info = None
        self.lr = None
        self.model = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.transform = None
        self.transform_post = None
        self.train_loader = None
        self.valid_loader = None
        self.inferer = None
        self.valid_metric = None

    def train_config(self, fl_ctx: FLContext):
        """MONAI traning configuration
        Here, we use a json to specify the needed parameters
        """

        # Load training configurations json
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        # Get the config_info
        self.lr = self.config_info["learning_rate"]
        cache_rate = self.config_info["cache_dataset"]
        dataset_path = self.config_info["dataset_base_dir"]
        datalist_path = self.config_info["datalist_json_path"]
        # Set the training-related context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = DiceLoss(sigmoid=True)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2),
            transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
            transforms.Resize((256, 256)),
            transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
        )
        traindata = ImageTextContrastiveDataset(datalist_path=datalist_path, dataset_path=dataset_path,
                                                imgtransform=transform, client_id = self.client_id)
        self.log_info(
            fl_ctx,
            f"Training Size: {len(traindata)}, Validation Size: {len()}",
        )
        train_collate_fn = ImageTextContrastiveCollator()

        train_dataloader = DataLoader(traindata,
                                 batch_size=10,
                                 collate_fn=train_collate_fn,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=1,
                                 )
        cls_prompts = generate_chexpert_class_prompts(n=10)
        val_data = ZeroShotImageDataset(['chexpert_5x200'],
                                        class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                        dataset_path= dataset_path)
        val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
                                               mode='multiclass')
        val_dataloader = DataLoader(val_data,
                                     batch_size=20,
                                     collate_fn=val_collate_fn,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1,
                                     )
        self.train_loader = train_dataloader
        self.valid_loader = val_dataloader
