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
import torch

from helpers.pt_fedsm import PTFedSMHelper

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal

from medclip.losses import ImageTextContrastiveLoss


class SupervisedPTFedSMHelper(PTFedSMHelper):
    """Helper to be used with FedSM components under supervised training specs"""

    def __init__(
        self,
        person_model,
        select_model,
        person_criterion,
        select_criterion,
        person_optimizer,
        select_optimizer,
        device,
        app_dir,
        person_model_epochs,
        select_model_epochs,
    ):
        super().__init__(
            person_model,
            select_model,
            person_criterion,
            select_criterion,
            person_optimizer,
            select_optimizer,
            device,
            app_dir,
            person_model_epochs,
            select_model_epochs,
        )

    def local_train_person(self, train_loader, abort_signal: Signal, writer, current_round):
        # Train personalized model, and keep track of curves
        for epoch in range(self.person_model_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            loss_model = ImageTextContrastiveLoss(self.person_model).to(self.device).to(dtype=torch.bfloat16)
            loss_model.train()
            epoch_len = len(train_loader)
            epoch_global = current_round * self.person_model_epochs + epoch
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                for key, value in batch_data.items():
                    if key != 'input_ids' and key != 'aug_input_ids':
                        batch_data[key] = value.to(dtype=torch.bfloat16)
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
                loss.backward()
                self.person_optimizer.step()
                self.person_optimizer.zero_grad()
                current_step = epoch_len * epoch_global + i
                writer.add_scalar("train_loss_personalized", loss.item(), current_step)
            torch.cuda.empty_cache()
