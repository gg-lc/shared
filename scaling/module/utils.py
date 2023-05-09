import logging
import os.path
import pprint

import numpy
import numpy as np
import pandas as pd
import time
import config
import sys
import cv2

sys.path.append('../protos')
from protos import service_pb2

root_dir = config.__file__.split('/')[:-1]
root_dir = '/'.join(root_dir)


def get_latency_ms(timestamp, now=None):
    if not now:
        now = get_timestamp()
    sec = now.seconds - timestamp.seconds
    mic = now.microseconds - timestamp.microseconds
    return sec * 1000 + mic / 1000


def get_timestamp():
    _t = time.time()
    return service_pb2.Timestamp(
        seconds=int(_t),
        microseconds=int(_t * 1000000) % 1000000
    )


def timestamp2float(timestamp: service_pb2.Timestamp):
    return timestamp.seconds + timestamp.microseconds * 0.0000001


def img2bytes(_img, ext='.jpg'):
    return b''
    return cv2.imencode(ext, _img)[1].tobytes()


def bytes2img(_bytes, flag=cv2.IMREAD_COLOR):
    return np.array([])
    arr = numpy.frombuffer(_bytes, numpy.uint8)
    return cv2.imdecode(arr, flag)


def set_logging_format(level=None):
    """ set logging format """
    # see logging.Formatter for more format
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] (%(module)s#%(lineno)d): %(message)s",
        datefmt="%H:%M:%S",
        level=config.LOGGING_LEVEL if level is None else level,
        # filename=config.LOG_FILE
    )


def load_trace() -> (numpy.ndarray, numpy.ndarray):
    """
    load trace and sub-request file (config.TRACES)
    :return: input_trace(intervals), [sub-1, sub-2, ...]
    """

    logging.info('Loading traces: {}'.format(config.WORKLOAD))
    trace_dir = r'../workload/trace/csv/{}.csv'
    sub_dir = r'../workload/sub/csv/{}.csv'

    interval = pd.read_csv(trace_dir.format(config.WORKLOAD[0]), header=None).to_numpy().flatten()
    min_len = len(interval)
    subs = []
    for i in range(1, len(config.WORKLOAD)):
        t = config.WORKLOAD[i]
        if '0' <= t <= '9':
            subs.append([int(t)] * min_len)
        else:
            subs.append(pd.read_csv(sub_dir.format(config.WORKLOAD[i]), header=None).to_numpy().flatten())
            min_len = min(min_len, len(subs[-1]))

    logging.info('Loaded traces: {} with len={}'.format(config.WORKLOAD, min_len))
    subs = np.round(np.transpose([_[:min_len] for _ in subs])).tolist()
    return interval, subs


def load_model_info(model_name):
    """
    load model cold start info
    :param model_name:
    :return: [load_model_time_s, first_infer_time_s]
    """
    profile = open(r'../data/profiles/{}.csv'.format(model_name)).readlines()
    return [float(_.split(':')[-1].split(',')[0]) for _ in profile[1:3]]


def load_first_batch_infer(model_name, aot=False):
    """
    load the first inference time under different batch size (ms)
    :param model_name: model name
    :param aot: whether to use AoT
    :return: dict: {bs:t, bs:t}
    """
    first_t = {}
    for line in open(r'../data/profiles/{}.csv'.format(model_name)).readlines():
        if '0' <= line[0] <= '9':
            line = line.split(',')
            first_t[int(line[0])] = float(line[2 if aot else 1])
    return first_t


def load_duration(model_name, batch_size=None):
    """
    load model duration (ms)
    :param model_name: model name
    :param batch_size: None by default. If not None, return
        a list of inference times under the specified batch size.
    :return: dict (batch_size is None) or list (batch_size is in [1, 2, 4, 8, 16, 32])
    """
    if batch_size is None:
        duration = {}
        for line in open(r'../data/profiles/{}.csv'.format(model_name)).readlines():
            if '0' <= line[0] <= '9':
                line = line.split(',')
                duration[int(line[0])] = float(line[3])
        return duration
    else:
        profile = r'../data/profiles/{}/{}_b{}.csv'.format(model_name, model_name, batch_size)
        return pd.read_csv(profile, header=None).to_numpy().flatten()


def load_profile(model, aot: bool):
    profile = {
        'model': model,  # model name
        'loading': .0,  # load model time (s)
        'first': {},  # first inference time (ms)
        'duration': {},  # batch duration (ms)
        'max_batch_size': 0
    }
    lines = open(r'../data/profiles/{}.csv'.format(model)).readlines()
    profile['loading'] = float(lines[1].split(':')[-1])

    for line in lines:
        if '0' <= line[0] <= '9':
            line = line.split(',')
            profile['first'][int(line[0])] = float(line[2 if aot else 1])
            profile['duration'][int(line[0])] = float(line[3])
            profile['max_batch_size'] = int(line[0])
    return profile


if __name__ == '__main__':
    pprint.pprint(load_profile('EfficientNetB7', True))
    pprint.pprint(load_profile('EfficientNetB7', False))
