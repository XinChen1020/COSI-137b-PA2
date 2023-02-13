# codeing: utf-8

import os
import logging
import torch

__all__ = ['logging_config', 'get_device']

def get_device():
    if torch.has_cuda:
        return torch.device("cuda")
    elif torch.has_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def logging_config(folder=None, name=None,
                   level=logging.INFO,
                   console_level=logging.INFO,
                   no_console=False):
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + '.log')
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder
