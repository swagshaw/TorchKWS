"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午4:57
@Author  : Yang "Jan" Xiao 
@Description : train
"""
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import *
from utils.utils import *

logger = logging.getLogger()


def get_dataloader_keyword(data_path, class_list, class_encoding, batch_size=1):
    """
    CL task protocol: keyword split.
    To get the GSC data and build the data loader from a list of keywords.
    """
    if len(class_list) != 0:
        train_filename = readlines(f"{data_path}/train.txt")
        valid_filename = readlines(f"{data_path}/valid.txt")
        train_dataset = SpeechCommandDataset(f"{data_path}/data", train_filename, True, class_list, class_encoding)
        valid_dataset = SpeechCommandDataset(f"{data_path}/data", valid_filename, False, class_list, class_encoding)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_dataloader, valid_dataloader
    else:
        raise ValueError("the class list is empty!")


class Trainer:
    """
    The KWS model training class.
    """

    def __init__(self, opt, model):
        self.opt = opt
        self.lr = opt.lr
        self.step = opt.step
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.model = model
        self.device, self.device_list = prepare_device(opt.gpu)
        # map the model weight to the device.
        self.model.to(self.device)
        # enable multi GPU training.
        if len(self.device_list) > 1:
            print(f">>>   Available GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = {
            "train_loss": 0.0, "train_accuracy": 0.0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0.0, "valid_accuracy": 0.0, "valid_total": 0, "valid_correct": 0}

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))

    def model_train(self, optimizer, train_dataloader, valid_dataloader):
        """
        Normal model training process, without modifying the loss function.
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        logger.info(f"[3] Training for {self.epoch} epochs...")
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.cuda(self.device)
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader), position=0):
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                # print((waveform.size,labels))
                optimizer.zero_grad()
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in enumerate(valid_dataloader):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
            scheduler.step()
            self.model_save()
            logger.info(
                f"Epoch {self.epo + 1}/{self.epoch} | train_loss {self.loss_name['train_loss']:.4f} "
                f"| train_acc {100 * self.loss_name['train_accuracy']:.4f} | "
                f"test_loss {self.loss_name['valid_loss']:.4f} "
                f"| test_acc {100 * self.loss_name['valid_accuracy']:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.4f}"
            )
        return self.loss_name
