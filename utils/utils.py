import numpy as np
import torch
import torch.nn as nn
import logging
import time


def Logger(log_file, name=None, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    if not logger.handlers:
        # handler = logging.FileHandler(log_file)
        # handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.setLevel(level)
        logger.addHandler(console)
        # logger.addHandler(handler)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def now_to_date(format_string="%Y-%m-%d[%H:%M:%S]"):
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def call_avg_coef(coefs):
    a1 = np.array(coefs)
    b1 = 0.5 * np.log((1+a1)/(1-a1))
    c1 = np.mean(b1)
    d1 = (np.e**(2*c1)-1) / (np.e**(2*c1)+1)
    return d1

def read_conf(str_conf, conf_type):
    str_conf = str_conf.split(',')
    confs = []
    for _, arg in enumerate(str_conf):
        tmp = None
        if conf_type[_] == 'int':
            tmp = int(arg)
        elif conf_type[_] == 'float':
            tmp = float(arg)
        elif conf_type[_] == 'arr_int':
            tmp = [int(x) for x in arg.split('-')]
        elif conf_type[_] == 'arr_float':
            tmp = [float(x) for x in arg.split('-')]
        elif conf_type[_] == 'str':
            tmp = arg
        else:
            print('arg type error!')
            exit(0)
        confs.append(tmp)
    if len(confs) == 1:
        confs = confs[0]
    return confs




