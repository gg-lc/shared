import logging
import multiprocessing
import os
import queue
import random
import threading
import time
import math
import cv2
import numpy as np
import setproctitle
import sortedcontainers
import tensorboardX

import config
import utils
import grpc
import matplotlib.pyplot as plt
from concurrent import futures
from protos import service_pb2, service_pb2_grpc

setproctitle.setproctitle('Client')
utils.set_logging_format()


# fixme: write configuration into tfevents
# todo: test gRPC throughput without inference
# todo: keep only one universal servicer in each module
#   (servicer puts requests to different queues depending on the type of request)
class ClientReceiver(service_pb2_grpc.InferenceServicer):
    def __init__(self, result_que: queue.Queue, msg_que: queue.Queue):
        # msg_que: receive TYPE_CONNECT[=1]
        self.queue = result_que
        self.msg_que = msg_que

    def SendRequest(self, request: service_pb2.Request, context):
        if request.type == service_pb2.TYPE_RESULT:
            request.latency.append(utils.get_latency_ms(request.begin))  # e2e latency
            self.queue.put(request)
            logging.debug('Receiver received request {}'.format(request.rid))
        elif request.type == service_pb2.TYPE_CONNECT:
            self.msg_que.put(request)
            logging.debug('Receiver received connection request ({})'.format(request.info['msg']))
        else:
            logging.error('Unknown message type: {}'.format(request.type))
            logging.error('Message: {}'.format(request))
        return service_pb2.Reply(flag=0)


class Receiver(multiprocessing.Process):
    def __init__(self, logdir: str, res_que: multiprocessing.Queue):
        super(Receiver, self).__init__()
        logging.info('Receiver started. logdir: {}'.format(logdir))
        self.que = res_que
        self.flag = threading.Event()

        # counter
        self.received = 0
        self.drop = 0
        self.done = 0
        self.effective = 0
        self.counter_lock = threading.Lock()
        self.drop_tmp = 0
        self.done_tmp = 0
        self.effective_tmp = 0

        # recorder
        self.latencies = []
        self.latencies_sort = sortedcontainers.SortedList()

        self.logdir = logdir
        self.writer = tensorboardX.SummaryWriter(
            logdir=os.path.join(logdir, 'client'),
            flush_secs=3,
            filename_suffix='.client',
        )
        cfg = '\n'.join(open(r'../config.py').readlines()).replace('#', '@')
        self.writer.add_text('config', cfg)

    def run(self) -> None:
        _receiver = threading.Thread(target=self.receiver)
        _logger = threading.Thread(target=self.logger)
        _receiver.start()
        _logger.start()
        _receiver.join()
        _logger.join()

    def receiver(self):
        while True:
            try:
                # get
                res: service_pb2.Request = self.que.get(timeout=5)
                lats = list(res.latency)
                # record
                with self.counter_lock:
                    self.received += 1
                    if res.info['done']:
                        self.done += 1
                        self.done_tmp += 1

                        self.latencies.append(lats[-1])
                        self.latencies_sort.add(lats[-1])
                        if lats[-1] <= config.MODULE_SLO[-1]:
                            self.effective += 1
                            self.effective_tmp += 1
                    else:
                        self.drop += 1
                        self.drop_tmp += 1
                # output
                ss = '{:.2f}'.format(lats[0])
                for i in range(1, len(lats)):
                    ss += ', {:.2f}'.format(lats[i])
                if random.randint(0, 100) == 1:
                    print('req-{}: [{}]ms'.format(res.rid, ss))
            except queue.Empty:
                self.flag.set()
                logging.warning('Finished receiving requests.')
                break
        logging.warning('Receiver-Receiver exited...')

    def logger(self):
        percentages = sorted([99.9, 99, 98, 95, 90, 85, 80, 1])  # todo: hyperparameter

        while not self.flag.is_set() or not self.que.empty():
            if self.received == 0:
                continue
            deadline = time.time() + config.LOGGING_INTERVAL * 0.001
            with self.counter_lock:  # snapshot
                received = self.received
                done, done_tmp = self.done, self.done_tmp
                drop, drop_tmp = self.drop, self.drop_tmp
                effect, effect_tmp = self.effective, self.effective_tmp
                self.drop_tmp = 0
                self.done_tmp = 0
                self.effective_tmp = 0

            # logging
            if len(self.latencies_sort) > 0:
                lat_percent = [self.latencies_sort[max(0, math.ceil(p / 100 * done) - 1)] for p in percentages]
                msg, scalars = '', {}
                for i, lat, percent in zip(range(len(percentages)), lat_percent, percentages):
                    msg += '| P{:4s}= {:7.2f}ms'.format(str(percent), lat) + (' |\n' if i % 4 == 3 else '  ')
                    if percent not in [1, 85, 98]:  # exclude
                        scalars['P{}'.format(percent)] = lat
                logging.warning('daemon ' + '-' * 40 + '\n'
                                + '| drop/done/recv: {:04d}/{:05d}/{:05d} '.format(drop, done, received)
                                + '| finished: {:6.2f}% '.format(100.0 * ((received - drop) / received))
                                + '| avg_lat_tmp: {:6.2f}ms |\n'.format(np.average(self.latencies[-done_tmp:]))
                                + msg + '-' * 80 + '\n')

                # tensorboard
                self.writer.add_scalar('client/avg_latency_tmp', np.average(self.latencies[-done_tmp:]), received)
                self.writer.add_scalars('client/percentile', scalars, received)
                if (done_tmp + drop_tmp) > 0:
                    self.writer.add_scalar('client/finish_rate_tmp', 100.0 * (done_tmp / (done_tmp + drop_tmp)),
                                           received)
                    self.writer.add_scalar('client/effective_rate_tmp', 100 * effect_tmp / (done_tmp + drop_tmp),
                                           received)
                self.writer.add_scalar('client/finish_rate', 100.0 * (done / received), received)
                self.writer.add_scalar('client/effective_rate', 100.0 * (effect / received), received)
                self.writer.flush()
            else:
                logging.warning('Client: no request done ... ')

            time.sleep(max(0, deadline - time.time()))

        # sorted latency

        latencies = np.array(self.latencies)
        latencies_sort = np.array(self.latencies_sort)
        np.savetxt(os.path.join(self.logdir, 'latencies.csv'), latencies, fmt='%.2f')
        np.savetxt(os.path.join(self.logdir, 'latencies_sorted.csv'), latencies_sort, fmt='%.2f')
        x = range(len(latencies))

        # plot
        plt.figure(figsize=(12, 6))
        plt.plot(x, latencies, label='latency')
        plt.plot(x, latencies_sort, label='latency (sort)', linewidth=2, color='red')
        plt.xlabel('request number')
        plt.ylabel('ms')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.logdir, 'latencies.png'))
        img = cv2.imread(os.path.join(self.logdir, 'latencies.png')).transpose(2, 0, 1)
        self.writer.add_image('latencies', img)

        self.writer.close()
        logging.warning('Receiver-Logger exited...')


class Client(multiprocessing.Process):
    def __init__(self, sample_img: str,
                 # network
                 ftd_port: str or int, ftd_ip: str, local_port: str or int, local_ip: str = '[::]',
                 # logging
                 logname: str = 'test', logdir: str = 'test'):
        super().__init__()
        # workload
        self.trace, self.subs = utils.load_trace()
        if config.WORKLOAD_SLICE is not None:
            n = len(self.trace)
            s = max(0, int(n * config.WORKLOAD_SLICE[0]))
            t = min(n, int(n * config.WORKLOAD_SLICE[1]))
            self.trace = self.trace[s:t]
            self.subs = self.subs[s:t]

        # request
        self.base_img = cv2.imread(sample_img)
        self.base_bytes = utils.img2bytes(self.base_img)

        # logging
        self.writer = None
        self.logdir = os.path.join(r'../log', logdir, logname + time.strftime('-%y%m%d.%H%M%S'))
        os.makedirs(self.logdir)

        # network
        self.ftd_address = ftd_ip + ':' + str(ftd_port)
        self.local_port = str(local_port)
        self.local_address = local_ip + ':' + str(local_port)

        # grpc
        # todo: whether the thread pool limits the system throughput?
        #   https://grpc.github.io/grpc/python/grpc.html#create-server
        self.result_que = multiprocessing.Queue()
        self.msg_que = queue.Queue()
        self.server = None
        self.stub = None

    def run(self):
        # create module
        _receiver = Receiver(self.logdir, res_que=self.result_que)
        _sender = threading.Thread(target=self.sender)

        os.system('cp ../config.py {}'.format(os.path.join(self.logdir, 'config.py')))
        # exit(0)

        # logging
        self.writer = tensorboardX.SummaryWriter(
            logdir=os.path.join(self.logdir, 'client'),
            flush_secs=3,
            filename_suffix='.client',
        )

        # start client
        self.make_connection()
        _receiver.start()
        time.sleep(3)
        _sender.start()

        # wait for stop
        logging.info('wait for the client to finish running')
        _sender.join()
        _receiver.join()
        logging.warning('client task finished.')

    # function
    def make_connection(self):
        # create server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
        servicer = ClientReceiver(self.result_que, self.msg_que)
        service_pb2_grpc.add_InferenceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port('[::]:' + self.local_port)
        self.server.start()
        logging.info('Receiver started at {}'.format(self.local_address))

        # connect to frontend
        channel = grpc.insecure_channel(target=self.ftd_address)
        self.stub = service_pb2_grpc.InferenceStub(channel)
        logging.info('Connected to frontend {}'.format(self.ftd_address))

        # send configuration (hello)
        self.stub.SendRequest(service_pb2.Request(
            type=service_pb2.TYPE_CONNECT,
            info={'msg': 1},
            msg={'logdir': self.logdir, 'client_addr': self.local_address},
            bytes=self.base_bytes
        ))

        # receive connection msg (ACK)
        connection: service_pb2.Request = self.msg_que.get()  # TYPE_CONNECT
        if connection.type == service_pb2.TYPE_CONNECT and connection.info['msg'] == 1:
            logging.info('Worker received connection request from Frontend, start running...')
        else:
            logging.error('Unknown operation: {}'.format(connection))
            raise ConnectionRefusedError

    def sender(self):
        def generate_request(_rid, _sub, data=None):
            req = service_pb2.Request(
                type=service_pb2.TYPE_REQUEST,
                rid=_rid, addr=self.local_address,
                bytes=utils.img2bytes(data) if data else self.base_bytes,
                begin=utils.get_timestamp()
            )
            for s in _sub:
                req.sub.append(s)
            return req

        # send request
        deadline = time.time()  # accurate timer
        np.random.seed(config.RAND_SEED)
        fluctuation = .0
        for rid, gap, sub in zip(range(len(self.trace)), self.trace, self.subs):
            gap *= config.GAP_FACTOR
            # fluctuation
            if rid % config.CHANGE_GAP == 0:
                fluctuation = np.random.normal(loc=0, scale=config.FLUCTUATION_RANGE / 2)
            request = generate_request(rid, sub)
            self.stub.SendRequest(request)
            logging.debug('Client sent request-{}({})'.format(rid, sub))
            deadline += (gap + fluctuation) * 0.001
            time.sleep(max(.0, deadline - time.time()))

        # disconnect
        time.sleep(10)
        self.stub.SendRequest(service_pb2.Request(
            type=service_pb2.TYPE_CONNECT,
            info={'msg': 0}
        ))
        logging.warning('Finished sending requests')


if __name__ == '__main__':
    client = Client(
        sample_img=r'../data/dataset/M-30.jpg',
        # logging
        logname=config.LOGNAME,
        logdir=config.LOGDIR,
        # network
        ftd_ip=config.FRONTEND_IP[0],
        ftd_port=config.FRONTEND_PORT[0],
        local_ip=config.CLIENT_IP,
        local_port=config.CLIENT_PORT,
    )
    client.start()
