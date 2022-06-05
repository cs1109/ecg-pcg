# torch
from pickle import FALSE
from numpy.lib.npyio import load
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
# model
from model.models import *
# utils
import os
import time
import config
import random
import logging
import numpy as np
from config import *
from pathlib import Path
from utils import utils


def normalize(x):
    return x/np.max(np.abs(x), axis=-1)[:, :, np.newaxis]

def dic2info(**args):
    info_string = ""
    for k,v in args.items():
        if len(np.shape(v)) == 0:
            if type(v) in [np.float64, np.float32]:
                info_string += (" - "+k+": {:.4f}".format(v))
            else:
                info_string += (" - "+k+": "+str(v))
        elif len(np.shape(v)) == 1:
            if len(v) > 3:
                continue
            elif  type(v[0]) in [np.float64, np.float32]:
                info_string += (" - "+k+": "+" ".join(["{:.4f}".format(vv) for vv in v]))
            else:
                info_string += (" - "+k+": "+str(v))
        else:
            continue
    return info_string

class Solver():
    def __init__(self, model, epoch_num, batch_size, learning_rate, reg_par, save_name, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=config.decay_rate)
        self.device = device
        self.save_name = save_name
        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler(OUT_DIR+"/ECG-PCG.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(self.optimizer)

    def set_train_set(self, ecg, pcg, y_list):
        self.train_set = [ecg, pcg, y_list]

    def set_test_set(self, ecg, pcg, y_list):
        self.test_set = [ecg, pcg, y_list]

    def fit(self):
        dset = TensorDataset(*self.train_set)
        dloader = DataLoader(dset, batch_size=self.batch_size, num_workers=10, shuffle=True)
        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            for data in dloader:
                self.optimizer.zero_grad()
                batch_ecg = data[0].cpu()
                batch_pcg = data[1].cpu()
                batch_y = data[2].cpu()
                y_pred, loss, _ = self.model(batch_ecg, batch_pcg, batch_y)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            train_loss = np.mean(train_losses)
            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f} - lr: {:.8f}"
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(epoch + 1, self.epoch_num, epoch_time, train_loss, self.optimizer.param_groups[0]["lr"]))
            if (epoch+1)%eval_epoch == 0:
                loss, score_train, att_pcg_ecg_list = self.test("train")
                loss, score_test, att_pcg_ecg_list = self.test("test")
                print("val loss: %.4f"%loss)
                score_train = np.concatenate([yy.cpu().numpy() for yy in score_train])
                score_test = np.concatenate([yy.cpu().numpy() for yy in score_test])
                acc_all, _ = utils.result_metrics(score_train, score_test, self.train_set[2].numpy().argmax(axis=1).reshape(-1,1), self.test_set[2].numpy().argmax(axis=1).reshape(-1,1), method=config.mod_name)
                info_string = "Calculate metrics"+dic2info(**acc_all)
                if acc_all["auc_test"][0] > 0.9:
                    torch.save(self.model.state_dict(),"_".join(self.save_name.split("_")[:-1]+["%03d"%(epoch+1)])+".model")
                    os.system("echo %03d %03d %.4f %.4f>> candidate.log"%(config.random_seed, epoch, acc_all["auc_test"][0],  acc_all["f1_test"][0]))
                self.logger.info(info_string)
    def test(self, set="test"):
        if set == "test":
            losses, outputs_list, att_pcg_ecg_list = self._get_outputs(*self.test_set)
        elif set == "train":
            losses, outputs_list, att_pcg_ecg_list = self._get_outputs(*self.train_set)
        return np.mean(losses), outputs_list, att_pcg_ecg_list

    def _get_outputs(self, ecg, pcg, y_list):
        with torch.no_grad():
            self.model.eval()
            dset = TensorDataset(ecg, pcg, y_list)
            dloader = DataLoader(dset, batch_size=self.batch_size, num_workers=10, shuffle=False)
            losses = []
            outputs_list = []
            att_pcg_ecg_list = []
            for data in dloader:
                batch_ecg = data[0].cpu()
                batch_pcg = data[1].cpu()
                batch_y = data[2].cpu()
                y_pred, loss, w_pcg_ecg = self.model(batch_ecg, batch_pcg, batch_y)
                att_pcg_ecg_list.append(w_pcg_ecg)
                outputs_list.append(y_pred.clone().detach())
                losses.append(loss.item())
        return losses, outputs_list, att_pcg_ecg_list


def main():
    assert not (config.load_model==False and config.testing==True)
    # load dataset
    to_file_train = '{}/{}'.format(DATASET_DIR, ftrain)
    if config.testing:
        to_file_test = '{}/{}'.format(DATASET_DIR, ftest)
    else:
        to_file_test = '{}/{}'.format(DATASET_DIR, fval)
    data_train_dict = np.load(to_file_train)
    data_test_dict = np.load(to_file_test)
    ecg_train_ = normalize(data_train_dict["ecg"][:, np.newaxis, :])
    ecg_test_= normalize(data_test_dict["ecg"][:, np.newaxis, :])
    pcg_train_ = normalize(data_train_dict["pcg"][:, np.newaxis, :])
    pcg_test_ = normalize(data_test_dict["pcg"][:, np.newaxis, :])
    labels_train = data_train_dict["labels"][:, np.newaxis].astype("int64")
    labels_test = data_test_dict["labels"][:, np.newaxis].astype("int64")
    utils.count_data(labels_train)
    utils.count_data(labels_test)
    ecg_train = torch.tensor(ecg_train_, dtype=torch.float32)
    pcg_train = torch.tensor(pcg_train_, dtype=torch.float32)
    labels_train_ = torch.tensor(labels_train, dtype=torch.int64)
    labels_train_ = torch.zeros(len(labels_train_), n_classes).scatter_(1, labels_train_, 1)
    ecg_test = torch.tensor(ecg_test_, dtype=torch.float32)
    pcg_test = torch.tensor(pcg_test_, dtype=torch.float32)
    labels_test_ = torch.tensor(labels_test, dtype=torch.int64)
    labels_test_ = torch.zeros(len(labels_test_), n_classes).scatter_(1, labels_test_, 1)
    save_name = "%s/%s_%s_%03d"%(OUT_DIR, ftrain.split(".")[0], config.mod_name, config.epoch_num)
    model = mod_list[config.mod_name](n_classes=n_classes).cpu()
    solver = Solver(model, config.epoch_num, batch_size, learning_rate, reg_par, save_name)
    solver.set_train_set(ecg_train, pcg_train, labels_train_)
    solver.set_test_set(ecg_test, pcg_test, labels_test_)
    if config.load_model:
        model.load_state_dict(torch.load(save_name+".model"))
    else:
        solver.fit()
        torch.save(model.state_dict(),save_name+".model")
    loss, score_train, att_pcg_ecg_list = solver.test("train")
    loss, score_test, att_pcg_ecg_list = solver.test("test")
    score_train = np.concatenate([yy.cpu().numpy() for yy in score_train])
    score_test = np.concatenate([yy.cpu().numpy() for yy in score_test])
    acc_all, pred_test = utils.result_metrics(score_train, score_test, labels_train, labels_test, method=config.mod_name, is_print=True)
    if not config.testing:
        np.savez(save_name+".npz", w_ecg_pcg = np.concatenate([att.cpu().detach().numpy() for att in att_pcg_ecg_list]), label=labels_test, res=(pred_test==labels_test[:,0]), **acc_all)
    return acc_all
    print("finished.")

if __name__ == "__main__":
    fix = "raw"
    regex = "train_%s_%s_*.model"%(fix, mod_name)
    root = Path("./output")
    filename = root.joinpath(regex)
    config.load_model=True
    config.testing=False
    res = main()
    config.threshold = res["threshold"]
    config.load_model=True
    config.testing=True
    config.mod_name = "RTNet"
    acc1 = main()
