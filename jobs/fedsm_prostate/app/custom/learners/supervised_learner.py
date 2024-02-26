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

from abc import abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from medclip.losses import ImageTextContrastiveLoss
from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.evaluator import Evaluator

class SupervisedLearner(Learner):
    def __init__(
        self,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Simple Supervised Trainer.
            This provides the basic functionality of a local learner: perform before-train validation on
            global model at the beginning of each round, perform local training, and send the updated weights.
            No model will be saved locally, tensorboard record for local loss and global model validation score.
            Enabled FedAvg

        Args:
            train_config_filename: directory of config_3 file.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.best_metric = 0.0
        self.client_id = None
        self.writer = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when a run starts, this is where the actual settings get initialized for trainer

        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # set local tensorboard writer for local validation score of global model
        self.writer = SummaryWriter(app_dir)
        # set the training-related contexts, this is task-specific
        self.train_config(fl_ctx)

    @abstractmethod
    def train_config(self, fl_ctx: FLContext):
        """Traning configurations customized to individual tasks
        This can be specified / loaded in any ways
        as long as they are made available for further training and validation
        some potential items include but not limited to:
        self.lr
        self.model
        self.device
        self.optimizer
        self.criterion
        self.transform_train
        self.transform_valid
        self.transform_post
        self.train_loader
        self.valid_loader
        self.inferer
        self.valid_metric
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(
        self,
        fl_ctx,
        train_loader,
        abort_signal: Signal,
        current_round,
    ):
        """Typical training logic
        Total local epochs: self.aggregation_epochs
        Load data pairs from train_loader: image / label
        Compute outputs with self.model
        Compute loss with self.criterion
        Update model
        """
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            loss_model = ImageTextContrastiveLoss(self.model).to(self.device)
            loss_model.train()
            epoch_len = len(train_loader)
            epoch_global = current_round * self.aggregation_epochs + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                current_step = epoch_len * epoch_global + i
                self.writer.add_scalar("train_loss", loss.item(), current_step)

    def local_valid(
        self,
        model,
        valid_loader,
        abort_signal: Signal,
        tb_id=None,
        current_round=None,
    ):
        """Typical validation logic
        Load data pairs from train_loader: image / label
        Compute outputs with self.model
        Perform post transform (binarization, etc.)
        Compute evaluation metric with self.valid_metric
        Add score to tensorboard record with specified id
        """
        medclip_clf = PromptClassifier(model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader = valid_loader,
            mode='multiclass',
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        # tensorboard record id, add to record if provided
        if tb_id:
            self.writer.add_scalar(tb_id, metric, current_round)
        return metric

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Typical training task pipeline with potential HE functionality
        Get global model weights (potentially with HE)
        Local training
        Return updated weights (model_diff)
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Typical validation task pipeline with potential HE functionality
        Get global model weights (potentially with HE)
        Validation on local data
        Return validation score
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)

        # validation on global model
        model_owner = "global_model"

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        # before_train_validate only, can extend to other validate types
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.local_valid(
                self.model,
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_global_model",
                current_round=current_round,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_global_model ({model_owner}): {global_metric:.4f}")
            # validation metrics will be averaged with weights at server end for best model record
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_metric},
                meta={},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
