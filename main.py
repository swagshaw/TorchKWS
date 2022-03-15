"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午6:07
@Author  : Yang "Jan" Xiao 
@Description : main
"""
import os
import logging.config
import time
import argparse
from utils.utils import *
from utils.train_utils import *
from train import Trainer, get_dataloader_keyword

if __name__ == "__main__":
    def options():
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=256, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=1, type=int, help="Number of GPU device")
        parser.add_argument("--root", default="./dataset", type=str, help="The path of dataset")
        parser.add_argument("--dataset", default="gsc", help="The name of the data set")
        parser.add_argument("--model", default="bcresnet8", type=str, help="models")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="weight", type=str, help="The save name")
        args = parser.parse_args()
        return args


    parameters = options()

    """
    Data 
    """
    class_list = ["yes", "no", "nine", "three", "bed", "up", "down", "wow", "happy", "four",
                  "stop", "go", "dog", "cat", "five", "tree", "one", "eight", "left", "right",
                  "bird", "seven", "six", "two", "marvin", "on", "sheila", "off", "house", "zero"]
    class_encoding = {category: index for index, category in enumerate(class_list)}

    """
    Logger 
    """
    save_path = f"{parameters.dataset}/{parameters.model}_lr{parameters.lr}_epoch{parameters.epoch}"
    logging.config.fileConfig("./logging.conf")
    logger = logging.getLogger()
    os.makedirs(f"logs/{parameters.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(f"[1] Select a KWS dataset ({parameters.dataset})")
    """
    Model 
    """
    model = select_model(parameters.model, len(class_list))
    logger.info(f"[2] Select a KWS model ({parameters.model})")
    optimizer, _ = select_optimizer("adam", parameters.lr, model)
    """
    Train 
    """
    data_path = os.path.join(parameters.root, parameters.dataset)
    train_loader, test_loader = get_dataloader_keyword(data_path, class_list,
                                                       class_encoding, parameters.batch)
    start_time = time.time()
    result = Trainer(parameters, model).model_train(optimizer=optimizer, train_dataloader=train_loader,
                                                    valid_dataloader=test_loader)

    """
    Summary
    """
    # Total time (T)
    duration = time.time() - start_time
    logger.info(f"======== Summary =======")
    logger.info(f"{parameters.model} parameters: {parameter_number(model)}")
    logger.info(f"Total time {duration}, Avg: {duration / parameters.epoch}s")
    logger.info(
        f"train_loss {result['train_loss']:.4f} "
        f"| train_acc {100 * result['train_accuracy']:.4f}")
    logger.info(
        f"test_loss {result['valid_loss']:.4f} "
        f"| test_acc {100 * result['valid_accuracy']:.4f}"
    )
