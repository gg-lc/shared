import os
import math
import time
import grpc

import config
import utils
import queue
import logging
import threading
import numpy as np
import setproctitle
import tensorboardX
import multiprocessing
from config import *
from predict import predictor
from collections import deque
from concurrent import futures
from protos import service_pb2, service_pb2_grpc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

utils.set_logging_format()


class FrontendReceiver(service_pb2_grpc.InferenceServicer):
    def __init__(self, req_queue, msg_queue, interval_queue, sf_que, step: list):
        self.request_queue = req_queue
        self.msg_queue = msg_queue  # connect message (connect / disconnect)
        self.interval_queue = interval_queue  # request interval -> request rate estimation
        self.sf_que = sf_que  # record the number of sub-request in each request -> update scaling factor
        self.last_receive = None
        self.step = step  # todo: change the list to Semaphore (Servicer.release -> Frontend.monitor.acquire -> step+=1)

    def SendRequest(self, request: service_pb2.Request, context):
        if request.type in [service_pb2.TYPE_CONNECT, service_pb2.TYPE_MESSAGE,
                            service_pb2.TYPE_HEARTBEAT, service_pb2.TYPE_SCALING,
                            service_pb2.TYPE_CONTROL]:
            self.msg_queue.put(request)
            logging.debug('Servicer received a message ({})'.format(REQUEST_TYPE_NAME[request.type]))
        elif request.type == service_pb2.TYPE_REQUEST:
            request.latency.append(utils.get_latency_ms(request.begin))
            self.request_queue.put(request)
            self.step[0] += 1
            if self.last_receive:
                self.interval_queue.put(time.time() - self.last_receive)
                if len(request.sub):
                    self.sf_que.put(request.sub[0])
                else:
                    self.sf_que.put(0)
            self.last_receive = time.time()
            logging.debug('Servicer received request {}({})'.format(request.rid, request.scaling_factor))
        else:
            logging.error('Unknown message type: {}'.format(request.type))
            logging.error('Message: {}'.format(request))
        return service_pb2.Reply(flag=0)


class Logger(multiprocessing.Process):
    def __init__(self, fid: int, logdir: str, suffix: str, duration: dict,
                 stat_que: multiprocessing.Queue):
        super(Logger, self).__init__()
        self.id = fid
        self.que = stat_que
        self.duration = duration
        self.writer = tensorboardX.SummaryWriter(
            logdir=logdir,
            flush_secs=2,
            filename_suffix=suffix
        )
        logging.info('Frontend-{} logdir: {}'.format(fid, logdir))

    def run(self):
        _need, _offer = .0, .0
        _cost = .0
        while True:
            try:
                stat: dict = self.que.get(timeout=5)
                if self.que.qsize() > 1:
                    logging.error('Frontend-{}.Logger cannot deal with the log data!!!'.format(self.id))
            except queue.Empty:
                break
            # info
            step, batch_size = stat['step'], stat['batch_size']
            # worker
            running, booting, idle = stat['running'], stat['booting'], stat['idle']
            # rate
            actual, predict = stat['rate_actual'], stat['rate_predict']
            sf_actual, sf_predict = stat['sf_actual'], stat['sf_predict']
            handle = running * batch_size * 1000 / self.duration[batch_size]
            qsize_sum, qsize_max = stat['qsize_sum'], stat['qsize_max']
            # utilization
            _need += actual
            _offer += handle
            util, extra = _need / _offer, (_offer - _need) / _need
            _cost += (running * NORMAL_COST + idle * IDLE_COST) * (LOGGING_INTERVAL / 1000)

            # batch_size & worker num
            self.writer.add_scalar('frontend-{}/batch_size'.format(self.id), batch_size, step)
            self.writer.add_scalar('frontend-{}/qsize_sum'.format(self.id), qsize_sum, step)
            self.writer.add_scalar('frontend-{}/qsize_max'.format(self.id), qsize_max, step)
            self.writer.add_scalars('frontend-{}/worker_num'.format(self.id), {
                'idle': idle, 'running': running, 'booting': booting}, step)
            # rate
            self.writer.add_scalars('frontend-{}/rate'.format(self.id), {
                'actual': actual, 'predict': predict, 'capability': handle}, step)
            # rate_pred
            self.writer.add_scalars('frontend-{}/rate_pred'.format(self.id),
                                    {'actual': actual}, step, walltime=time.time())
            self.writer.add_scalars('frontend-{}/rate_pred'.format(self.id),
                                    {'predict': predict}, step, walltime=time.time() + WORKER_BOOT_TIME * 0.001)
            # sf_pred
            self.writer.add_scalars('frontend-{}/scaling_factor_pred'.format(self.id),
                                    {'actual': sf_actual}, step, walltime=time.time())
            self.writer.add_scalars('frontend-{}/scaling_factor_pred'.format(self.id),
                                    {'predict': sf_predict}, step, walltime=time.time() + WORKER_BOOT_TIME * 0.001)
            # utilization & cost
            self.writer.add_scalars('frontend-{}/util'.format(self.id), {
                'utilization': util, 'extra_resource': extra}, step)
            self.writer.add_scalars('frontend-{}/cost'.format(self.id), {'cost': _cost}, global_step=step)

        time.sleep(3)
        self.writer.flush()
        self.writer.close()
        logging.warning('Frontend-{}.Logger exited...'.format(self.id))


class Frontend(multiprocessing.Process):
    def __init__(self, ip: str, port: str or int, fid: int):
        # basic info
        super().__init__()
        self.id = fid
        self.port = str(port)
        self.address = ip + ':' + str(port)
        self.dummy_bytes = None

        # logging
        self.logdir = None
        self.step = [0]

        # client info & communication
        self.client_addr = None
        self.client_stub = None

        # worker info
        self.max_worker = WORKER_NUM[self.id]
        self.worker_processes = WORKER_PROCESSES[self.id]
        self.worker_ips = WORKER_IP[self.id]
        self.worker_ports = WORKER_PORT[self.id]

        # worker control
        self.control_lock = threading.Lock()  # batch_sizes & worker_stats & counter
        self.running_worker = None  # number of running worker
        self.booting_worker = None
        self.idle_worker = None
        self.worker_stats = None
        self.batch_size = None
        self.worker_stubs = []  # list, stubs used to control (communicate with) workers.

        # dispatcher
        self.current_worker = -1  # The worker to which the current batch is sent.
        self.load_balance = config.LOAD_BALANCE
        self.worker_qsize = None

        # config info
        self.model_name = MODEL[self.id]
        self.module_slo = MODULE_SLO[self.id]
        self.slo = self.module_slo
        if self.id > 0:
            self.slo -= MODULE_SLO[self.id - 1]
        self.profile = utils.load_profile(self.model_name, AOT_TUNER)
        self.duration = self.profile['duration']
        self.scaling_lock = threading.Lock()
        self.rate_handle = None  # maximum allowable request rate
        self.rate_current = INIT_QPS[self.id]
        self.rate_future = INIT_QPS[self.id] * (1 + EXTRA_CAPACITY)

        # next_frontend & scaling factor
        self.sf_future = INIT_SF[self.id]
        self.sf_current = INIT_SF[self.id]
        self.last_cascade_scaling = None
        self.next_frontend = None
        if self.id + 1 < FRONTEND_NUM:
            self.next_frontend = FRONTEND_IP[self.id + 1] + ':' + str(FRONTEND_PORT[self.id + 1])
        self.next_frontend_stub = None

        # scheduler
        self.timer = None  # timer for periodic scheduling
        self.predictor_s = None
        self.predictor_l = None

        # flag
        self.connected = False  # set to True after client connection

        # queue & grpc server
        self.request_queue = queue.Queue()
        self.msg_queue = queue.Queue()
        self.stat_queue = queue.Queue()
        self.interval_queue = queue.Queue()
        self.sf_queue = queue.Queue()
        self.server = None

    # function
    def run(self) -> None:
        setproctitle.setproctitle('Frontend-{}'.format(self.id))
        self.predictor_s = predictor.PredictorLSTM(
            init_load=self.rate_current, model_path='predict/LSTM/lstm_wiki_3.h5',
            scaler_path='predict/LSTM/my_scaler.save', n_out=3)
        self.predictor_l = predictor.PredictorLSTM(
            init_load=self.rate_current, model_path='predict/LSTM/lstm_wiki_6.h5',
            scaler_path='predict/LSTM/my_scaler.save', n_out=6)

        # create gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
        servicer = FrontendReceiver(self.request_queue, self.msg_queue, self.interval_queue, self.sf_queue, self.step)
        service_pb2_grpc.add_InferenceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port('[::]:' + self.port)
        self.server.start()
        logging.info('Frontend-{} listening at {}'.format(self.id, self.address))

        # wait for client connection & initialization
        connection: service_pb2.Request = self.msg_queue.get()
        if not connection.info['msg'] == 1:
            raise grpc.RpcError
        self.logdir = connection.msg['logdir']
        self.client_addr = connection.msg['client_addr']
        self.dummy_bytes = connection.bytes
        setproctitle.setproctitle('Frontend-{}'.format(self.id))

        # first schedule
        self.batch_size = None  # find the maximum batch_size that could satisfy the SLO
        if DISPATCH == 'batching':
            for bs in reversed(sorted(self.duration.keys())):
                latency = self.duration[bs] + bs / self.rate_future * 1000 + OVERHEAD
                if latency <= self.slo:
                    self.batch_size = bs
                    break
        elif DISPATCH == 'round-robin':
            for bs in reversed(sorted(self.duration.keys())):
                latency = self.duration[bs] * 2 + OVERHEAD
                if latency <= self.slo:
                    self.batch_size = bs
                    break
        else:
            raise NotImplementedError
        if self.batch_size is None:
            logging.error('There\'s no batch size could satisfy the SLO, set batch size to {}...'.format(
                list(self.duration.keys())[0]
            ))
            self.batch_size = list(self.duration.keys())[0]
        throughput = self.batch_size * (1000 / self.duration[self.batch_size])
        self.running_worker = math.ceil(self.rate_future / throughput)
        self.booting_worker = 0
        self.idle_worker = 0
        self.worker_stats = [STAT_ON] * self.running_worker + [STAT_OFF] * (self.max_worker - self.running_worker)
        self.worker_qsize = [0] * self.max_worker
        self.rate_handle = self.running_worker * throughput

        # connect to workers & client
        self._make_connection()
        self.connected = True

        logging.debug({
            'config/frontend': 'IP: {}\n\nport: {}'.format(FRONTEND_IP[self.id], FRONTEND_PORT[self.id]),
            'config/worker': 'num: {}\n\nmin_num: {}\n\nIP: {}\n\nport: {}'.format(
                WORKER_NUM[self.id], WORKER_NUM_MIN[self.id], WORKER_IP[self.id], WORKER_PORT[self.id]),
            'workload': f'duration: {self.duration}\n\nslo: {self.slo}ms\n\nmodule_slo: {self.module_slo}ms',
            'frontend': f'schedule_interval_ms: {SCHEDULE_INTERVAL}\n\nextra capacity: {EXTRA_CAPACITY * 100}%'
                        f'cascade scaling {CASCADE_SCALING}\n\n'
                        f'cascade effective duration: {CASCADE_EFFECTIVE_DURATION}\n\n'
                        f'estimation_policy: {ESTIMATION_POLICY}\n\n'
                        f'estimation_interval: {INTERVAL_ESTIMATION}\n\n'
                        f'estimation_history_length_ms: {HISTORY_LENGTH}',
            'worker': 'Drop: {}\n\nDummy: {}\n\ncold start: {}\n\nheartbeat_interval: {}'.format(
                DROP, DUMMY, WORKER_BOOT_TIME, LOGGING_INTERVAL)
        })

        # create module and start running
        _qps_monitor = threading.Thread(target=self.qps_monitor)
        _scheduler = threading.Thread(target=self.scheduler)
        _logger = threading.Thread(target=self.logger)
        _broker = threading.Thread(target=self.broker)
        _dispatcher = threading.Thread(target=self.dispatcher)

        _qps_monitor.start()
        _scheduler.start()
        _logger.start()
        _broker.start()
        _dispatcher.start()
        logging.info('Frontend-{} start working...'.format(self.id))

        try:  # wait for disconnect / KeyboardInterrupt
            while self.connected:
                time.sleep(3)
        except KeyboardInterrupt:
            logging.warning('Frontend-{} caught KeyboardInterrupt, trying to stop...'.format(self.id))
            self.connected = False
        finally:
            logging.warning('Frontend-{} disconnected from client...'.format(self.id))
            _qps_monitor.join()
            _scheduler.join()
            _logger.join()
            _broker.join()
            _dispatcher.join()
        logging.warning('Frontend-{} stopped'.format(self.id))

    # function
    def _make_connection(self):
        # connect to next frontend (if needed)
        if self.next_frontend is not None:
            channel = grpc.insecure_channel(target=self.next_frontend)
            self.next_frontend_stub = service_pb2_grpc.InferenceStub(channel)
            logging.info('Frontend-{} connected to Frontend-{}'.format(self.id, 1))
            self.next_frontend_stub.SendRequest(service_pb2.Request(
                type=service_pb2.TYPE_CONNECT,
                info={'msg': 1},
                msg={'logdir': self.logdir, 'client_addr': self.client_addr},
                bytes=self.dummy_bytes
            ))

        # connect to workers
        for _, ip, port in zip(range(self.max_worker), self.worker_ips, self.worker_ports):
            wid = (self.id + 1) * 100 + _  # id: 1xx 2xx ...
            channel = grpc.insecure_channel(target=ip + ':' + str(port))
            self.worker_stubs.append(service_pb2_grpc.InferenceStub(channel))
            self.worker_stubs[-1].SendRequest(
                service_pb2.Request(
                    type=service_pb2.TYPE_CONNECT,
                    info={
                        'id': wid,
                        'msg': 1,
                        'slo': self.slo,
                        'module_slo': self.module_slo,
                        'stat': self.worker_stats[_],
                        'batch_size': self.batch_size,
                    },
                    msg={
                        'logdir': self.logdir,
                        'model_name': self.model_name,
                        'frontend_addr': self.address,
                        'client_addr': self.client_addr,
                        'next_frontend': self.next_frontend if self.next_frontend else ''
                    },
                    bytes=self.dummy_bytes
                )
            )
            logging.info('Frontend-{} connected to worker-{}'.format(self.id, wid))

        # reply a success msg to client
        if self.id == FRONTEND_NUM - 1:
            channel = grpc.insecure_channel(target=self.client_addr)
            self.client_stub = service_pb2_grpc.InferenceStub(channel)
            self.client_stub.SendRequest(service_pb2.Request(
                type=service_pb2.TYPE_CONNECT, info={'msg': 1}))
            logging.info('Frontend-{} connected to client'.format(self.id))

    # function
    def _update_worker_stat(self, wid: int, new: int):
        with self.control_lock:
            now = self.worker_stats[wid]
            if now == config.STAT_ON:  # ON -> *
                self.running_worker -= 1
            elif now == config.STAT_BOOT:  # BOOT -> *
                self.booting_worker -= 1
            elif now == config.STAT_IDLE:  # IDLE -> *
                self.idle_worker -= 1
            if new == config.STAT_ON:  # * -> ON
                self.running_worker += 1
            elif new == config.STAT_BOOT:  # * -> BOOT
                self.booting_worker += 1
            elif new == config.STAT_IDLE:
                self.idle_worker += 1
            self.worker_stats[wid] = new
            self.rate_handle = self.running_worker * self.batch_size * 1000 / self.duration[self.batch_size]

    # component
    def scheduler(self, cascade_target=None):
        """
        control the status of each worker & the batch_size of (each) worker
        based on [rate, future_rate, slo, duration, ...]
        """

        # todo: send the scaling factor only when the rate increases???

        def update_batch_size(target: int):
            """
            send the batch size to all workers
            todo: adjust to update the batch_size of specific worker
            """
            if self.batch_size == target:
                logging.debug('Target batch size == current batch size, no need to adjust')
                return
            self.batch_size = target
            logging.debug('Update batch size of all workers to {}'.format(target))
            request = service_pb2.Request(
                type=service_pb2.TYPE_CONTROL,
                info={'batch_size': target}
            )
            for stub in self.worker_stubs:
                stub.SendRequest(request)

        def scaling(target_num: int):
            """
            adjust the number of running worker to target_num.
            :param target_num: self.max_worker > target_num > WORKER_NUM_MIN
            """
            if target_num < WORKER_NUM_MIN[self.id] or target_num > self.max_worker:
                logging.error('Target number of workers {} invalid!'.format(target_num))
                target_num = max(target_num, WORKER_NUM_MIN[self.id])
                target_num = min(target_num, self.max_worker)
                logging.error('Change the target number of workers to {}'.format(target_num))

            # scaling up
            if target_num > self.running_worker + self.booting_worker:
                request = service_pb2.Request(
                    type=service_pb2.TYPE_CONTROL,
                    info={'stat': STAT_ON})
                for _ in range(self.max_worker):
                    if self.booting_worker + self.running_worker >= target_num:
                        break
                    if self.worker_stats[_] == STAT_IDLE:
                        logging.info('Frontend-{} try to launch Worker-{} (quick)'.format(self.id, _))
                        self._update_worker_stat(_, STAT_BOOT)
                        self.worker_stubs[_].SendRequest(request)
                for _ in range(self.max_worker):
                    if self.booting_worker + self.running_worker >= target_num:
                        break
                    if self.worker_stats[_] == STAT_OFF:
                        logging.debug('Frontend-{} try to launch Worker-{} (normal)'.format(self.id, _))
                        self._update_worker_stat(_, STAT_BOOT)
                        self.worker_stubs[_].SendRequest(request)
                logging.info('Frontend-{} scaling up workers to {}'.format(self.id, target_num))

            # scaling down
            elif target_num < self.running_worker + self.booting_worker:
                # do not reduce the number of worker if the time from last cascade scaling is too short...
                if self.last_cascade_scaling and \
                        time.time() < self.last_cascade_scaling + CASCADE_EFFECTIVE_DURATION * 0.001:
                    logging.warning('Frontend-{}: SCALING INTERVAL < CASCADE SCALING EFFECTIVE TIME. no scaling...')
                    return
                request = service_pb2.Request(
                    type=service_pb2.TYPE_CONTROL,
                    info={'stat': STAT_OFF})
                for _ in range(self.max_worker):
                    if self.booting_worker + self.running_worker <= target_num:
                        break
                    if self.worker_stats[_] == STAT_BOOT:
                        logging.debug('Frontend-{} shutdown booting Worker-{}'.format(self.id, _))
                        if TURN_OFF_DELAY:
                            self._update_worker_stat(_, STAT_IDLE)
                        else:
                            self._update_worker_stat(_, STAT_OFF)
                        self.worker_stubs[_].SendRequest(request)
                for _ in range(self.max_worker):
                    if self.booting_worker + self.running_worker <= target_num:
                        break
                    if self.worker_stats[_] == STAT_ON:
                        logging.debug('Frontend-{} shutdown running Worker-{}'.format(self.id, _))
                        if TURN_OFF_DELAY:
                            self._update_worker_stat(_, STAT_IDLE)
                        else:
                            self._update_worker_stat(_, STAT_OFF)
                        self.worker_stubs[_].SendRequest(request)
                logging.info('Frontend-{} scaling down workers to {}'.format(self.id, target_num))

            # target == running + booting
            else:
                logging.debug('Frontend-{} no need to scaling...'.format(self.id))
            stats = '['
            for _ in self.worker_stats:
                stats += str(_)
            stats += ']'
            logging.info('Frontend-{} worker stat: {}'.format(self.id, stats))

        def warmup(target_num: int):
            request = service_pb2.Request(
                type=service_pb2.TYPE_CONTROL,
                info={'stat': STAT_IDLE}
            )
            for _ in range(self.max_worker):
                if self.idle_worker >= target_num:
                    break
                if self.worker_stats[_] == STAT_OFF:
                    logging.info('Frontend-{} warmup worker-{}'.format(self.id, _))
                    # the warmed worker will be set to 'idle' directly after booting
                    # without stat_on -> cannot set the worker stat to stat_on here.
                    self._update_worker_stat(_, STAT_IDLE)
                    self.worker_stubs[_].SendRequest(request)
            if self.idle_worker < target_num:
                logging.warning('Frontend-{} no enough workers to warmup ({})'.format(self.id, target_num))
            logging.info('Frontend-{} warmed up #{} workers'.format(self.id, self.idle_worker))

        def cascade_scaling(_rate_future: int):
            if self.id == FRONTEND_NUM - 1 or not CASCADE_SCALING:
                return
            self.next_frontend_stub.SendRequest(service_pb2.Request(
                type=service_pb2.TYPE_SCALING,
                scaling_factor=_rate_future))

        self.scaling_lock.acquire()

        _start_schedule = time.time()
        logging.debug('Frontend-{} starts a schedule'.format(self.id))
        rate_handle, rate_current, rate_future = self.rate_handle, self.rate_current, self.rate_future
        slo, durations = self.slo, self.duration

        if cascade_target is not None:
            rate_future = max(rate_future, cascade_target)
            cascade_scaling(cascade_target * self.sf_future)
            self.last_cascade_scaling = time.time()
        elif CASCADE_SCALING:
            cascade_scaling(rate_future * self.sf_future)

        # target rate
        rate_target = rate_future * (1 + EXTRA_CAPACITY)
        warm_target = rate_target * (1 + WARMUP_RATIO)
        bs_target = None

        # step-1: choose the largest batch size that can
        if DISPATCH == 'batching':
            for bs in reversed(sorted(durations.keys())):
                # note: 20ms for communication and other overhead
                latency = durations[bs] + (bs - 1) / rate_target * 1000 + OVERHEAD
                if latency <= slo:
                    bs_target = bs
                    break
        elif DISPATCH == 'round-robin':
            for bs in reversed(sorted(durations.keys())):
                latency = durations[bs] * 2 + OVERHEAD
                if latency <= slo:
                    bs_target = bs
                    break
        else:
            raise NotImplementedError

        if bs_target is None:
            logging.warning('There\'s no batch size could satisfy the SLO, set batch size to {}...'.format(
                list(durations.keys())[0]
            ))
            bs_target = list(durations.keys())[0]

        # step-2: calculate the number of workers target
        throughput = 1000 / durations[bs_target] * bs_target
        worker_target = math.ceil(rate_target / throughput)
        warmup_num = math.ceil(warm_target / throughput) - worker_target

        # update batch size and scale the number of workers
        update_batch_size(bs_target)
        scaling(worker_target)
        if PRE_WARMUP:
            warmup(warmup_num)

        logging.info(f'Frontend-{self.id} scheduled: '
                     f'[rate_target={rate_target}, num_worker={worker_target}, '
                     f'batch_size={bs_target}; cascade={cascade_target}]')

        # todo: scaling factor
        #   note: the initial value of the scaling factor shouldn't be set manually
        # if rate_future > rate_handle -> send_scaling_factor(rate_future*scaling_factor)

        if self.connected:
            self.timer = threading.Timer(SCHEDULE_INTERVAL * 0.001 - (time.time() - _start_schedule), self.scheduler)
            self.timer.start()
        else:
            logging.warning('Frontend-{}: scheduler exited...'.format(self.id))
        self.scaling_lock.release()

    # component
    def broker(self):
        """
        receive msg from other module (heartbeat, scaling_factor, launch, connect[-1])
        1. receive & process scaling msg from other Frontend -> schedule immediately
        2. receive launch msg from worker -> lock & update stats (booting-1 running+1)
        """
        while self.connected or not self.msg_queue.empty():
            msg: service_pb2.Request = self.msg_queue.get()
            if msg.type == service_pb2.TYPE_HEARTBEAT:  # heartbeat from worker
                logging.debug('Worker-{} STAT={} Qsize={} SendR={} AvgLat={} BS={}'.format(
                    msg.info['id'], msg.info['stat'], msg.info['qsize'], msg.info['proc_rate'],
                    msg.info['avg_lat'], msg.info['batch_size']))
                # with self.control_lock:
                #     self.worker_qsize[msg.info['id'] % 100] = msg.info['qsize']
                logging.debug('Frontend-{} received heartbeat from Worker-{}'.format(self.id, msg.info['id']))
            elif msg.type == service_pb2.TYPE_SCALING:  # cascade scaling
                logging.info('Frontend-{} received cascade scaling request, future={}'.format(
                    self.id, msg.scaling_factor))
                if msg.scaling_factor > self.rate_handle:
                    logging.info('Frontend-{} Schedule trigger: handle={} < scaling={}'.format(
                        self.id, self.rate_handle, msg.scaling_factor))
                    self.timer.cancel()
                    self.scheduler(cascade_target=msg.scaling_factor)
            elif msg.type == service_pb2.TYPE_CONTROL and msg.info['stat'] == STAT_ON:  # worker launch
                wid = msg.info['id']
                if self.worker_stats[wid] == STAT_BOOT:
                    self._update_worker_stat(wid, STAT_ON)
                    logging.info('Frontend-{} launched worker-{}'.format(self.id, msg.info['id']))
                else:
                    logging.warning('Worker {}-{} #cur={} recv={} failed...'.format(
                        self.id, wid, self.worker_stats[wid], STAT_ON))
            elif msg.type == service_pb2.TYPE_CONTROL and msg.info['stat'] == STAT_OFF:  # worker shutdown
                wid = msg.info['id']
                if self.worker_stats[wid] == STAT_IDLE:
                    self._update_worker_stat(wid, STAT_OFF)
                    logging.info('Frontend-{} shutdown worker-{}'.format(self.id, msg.info['id']))
                else:
                    logging.warning('Worker {}-{} #cur={} recv={} failed...'.format(
                        self.id, wid, self.worker_stats[wid], STAT_OFF))
            elif msg.type == service_pb2.TYPE_CONNECT and msg.info['msg'] == 0:
                logging.warning('Frontend-{} received disconnect msg, ready to stop...'.format(self.id))
                self.connected = False
                if self.next_frontend_stub is not None:
                    self.next_frontend_stub.SendRequest(msg)
                for stub in self.worker_stubs:
                    stub.SendRequest(msg)
            else:
                logging.error('Frontend-{}: Unknown msg: {}'.format(self.id, msg))
        logging.warning('Frontend-{}: broker exited...'.format(self.id))

    # function
    def _predict(self, y: list or float, p: int) -> int:
        """
        Predict the future workload
        :param y: history data
        :param p: predict length
        :return:
        """
        if not PREDICT:
            if isinstance(y, list):
                return np.mean(y[-min(len(y), 3):])
            else:
                return y

        # todo: implement more policy (PR, MWA, LSTM...)
        # todo: verify the performance of different predict policy
        n = len(y)
        x = np.array(range(n))
        if ESTIMATION_POLICY == 'LR':
            # len(his)==100 pred=10/20 -> time~0.4ms
            # len(his)==500 pred=10/20 -> time~0.65ms
            x = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(np.array(range(n), n + p).reshape(-1, 1))
            return math.ceil(y_pred[-1])  # todo: return max or last?
        elif ESTIMATION_POLICY.startswith('PolyReg-'):
            # degree  history pred_len  time
            #   2       100     10/20   0.8ms
            #   3       100     10/20   0.9ms
            #   2       500     10/20   1.0ms
            #   3       500     10/20   1.1ms
            poly = PolynomialFeatures(degree=int(ESTIMATION_POLICY[-1]))
            x = poly.fit_transform(x.reshape(-1, 1))
            model = LinearRegression()
            model.fit(x, y)
            x_pred = poly.transform(np.array(range(n, n + p)).reshape(-1, 1))
            y_pred = model.predict(x_pred)
            # print(f'[###] {y} {p} {y_pred}')
            return max(1, math.ceil(y_pred[-1]))
        elif ESTIMATION_POLICY == 'envelope':
            window_interval = HISTORY_LENGTH
            window_size = n
            rate_max = -1
            while window_interval >= WORKER_BOOT_TIME:
                rate_max = max(rate_max, np.mean(y[-int(window_size):]))
                for d in range(1, int(n // window_size)):
                    rate_max = max(rate_max, np.mean(y[-int(window_size) * (d + 1):-int(window_size) * d]))
                window_interval /= 2
                window_size /= 2
            return rate_max
        else:
            raise NotImplementedError

    # method
    def _fluctuation(self, real, pred, n):
        d = np.array(real) - np.array(pred)
        dmin, dmax = np.min(d), np.max(d)
        d = np.sort(d)
        f = max(0, np.mean(d[int(n * (PERCENTILE - 0.1)):int(n * PERCENTILE)]))
        # f = max(0, d[int(n*PERCENTILE)])
        # logging.warning('Frontend-{} FLUC_arr={}[{}]'.format(self.id, d, int(n*PERCENTILE)))
        logging.warning('Frontend-{} FLUCTUATION = {}'.format(self.id, f))
        return f

    # component
    def qps_monitor(self) -> None:
        """
        Monitor the request rate and update self.qps
        Usage: call as thread!
        Method: use self.interval_queue to get intervals between each request,
                analyze intervals trend and calculate qps (at multi-scale).
        """
        history_len = math.ceil(HISTORY_LENGTH / INTERVAL_ESTIMATION)  # len of historical data
        predict_len = math.ceil(WORKER_BOOT_TIME / INTERVAL_ESTIMATION)  # len of predict sequence
        history = deque(iterable=[self.rate_current for _ in range(history_len)], maxlen=history_len)
        history_sf = deque(iterable=[self.sf_future for _ in range(history_len)], maxlen=history_len)

        # used for fluctuation CDF & RMSE
        real_workload = []
        pred_workload = []
        fluc_workload = []
        lstm_workload = []
        for _ in range(predict_len):  # delta
            pred_workload.append(0)
            lstm_workload.append(0)
            fluc_workload.append(0)
        for _ in range(10):
            fluc_workload.append(0)

        # predict & update & log
        # todo: multi-scale monitor
        deadline = time.time() + INTERVAL_ESTIMATION * 0.001  # make the estimator more accurate
        received_first_reqeust = False
        time.sleep(0.5)  # wait the rate to be stable...
        while self.connected:
            time.sleep(max(0, deadline - time.time()))
            que_size = self.interval_queue.qsize()  # snapshot -> more accurate
            if que_size:
                received_first_reqeust = True
            if que_size < 10 or not received_first_reqeust:  # scaling after receiving the first request
                continue
            received_first_reqeust = True

            # predict
            total = 0
            for _ in range(que_size):
                total += self.interval_queue.get()
            self.rate_current = que_size / total if que_size else 0  # current QPS
            self.rate_future = self._predict(list(history), predict_len)
            history.append(self.rate_current)

            # save
            pred_workload.append(self.rate_future)
            real_workload.append(self.rate_current)

            # lstm
            lstm_workload.append(self.predictor_s.predict(self.rate_current)[-1])
            if PREDICT and USE_LSTM:
                self.rate_future = lstm_workload[-1]

            # CDF
            if FLUCTUATION:
                if len(pred_workload) > 10 + predict_len:
                    n = min(history_len, len(pred_workload) - predict_len)
                    fluc = self._fluctuation(real_workload[-n:], pred_workload[-n:], n)
                    self.rate_future += fluc
                    fluc_workload.append(self.rate_future)
                else:
                    self.rate_future += self.rate_future * 0.05

            que_size = self.sf_queue.qsize()
            total = 0
            for _ in range(que_size):
                total += self.sf_queue.get()
            self.sf_current = total / que_size if que_size else 0
            self.sf_future = self._predict(list(history_sf), predict_len)

            # trigger scheduling
            if TRIGGER_SCHEDULING and (self.rate_handle < self.rate_future or self.rate_handle < self.rate_current):
                logging.info('Frontend-{} Schedule trigger: rate_predict > rate_handle'.format(self.id))
                self.timer.cancel()
                self.scheduler()

            history_sf.append(self.sf_current)

            logging.debug('Monitor: current rate={}, future rate={}'.format(self.rate_current, self.rate_future))
            deadline += INTERVAL_ESTIMATION * 0.001

        real_workload = np.array(real_workload)
        lstm_workload = np.array(lstm_workload)
        pred_workload = np.array(pred_workload)
        fluc_workload = np.array(fluc_workload)
        np.savetxt(os.path.join(self.logdir, 'WORKLOAD_LSTM-FTD{}.csv'.format(self.id)), lstm_workload, delimiter=',')
        np.savetxt(os.path.join(self.logdir, 'WORKLOAD_PRED-FTD{}.csv'.format(self.id)), pred_workload, delimiter=',')
        np.savetxt(os.path.join(self.logdir, 'WORKLOAD_REAL-FTD{}.csv'.format(self.id)), real_workload, delimiter=',')
        np.savetxt(os.path.join(self.logdir, 'WORKLOAD_FLUC-FTD{}.csv'.format(self.id)), fluc_workload, delimiter=',')

        n = min(len(real_workload), len(pred_workload), len(lstm_workload)) - predict_len
        d_lstm = np.array(real_workload[-n:]) - np.array(lstm_workload[-n:])
        rmse_lstm = np.sqrt(np.mean(d_lstm ** 2))
        d_pred = np.array(real_workload[-n:]) - np.array(pred_workload[-n:])
        rmse_pred = np.sqrt(np.mean(d_pred ** 2))
        logging.info('Frontend-{} RMSE-LSTM = {}'.format(self.id, rmse_lstm))
        logging.info('Frontend-{} RMSE-Poly = {}'.format(self.id, rmse_pred))
        with open(os.path.join(self.logdir, 'RMSE-FRONTEND-{}'.format(self.id)), 'w') as f:
            f.write('Frontend-{} RMSE-LSTM = {}'.format(self.id, rmse_lstm))
            f.write('Frontend-{} RMSE-Poly = {}'.format(self.id, rmse_pred))

        logging.warning('Frontend-{}: QPS monitor exited...'.format(self.id))

    # function
    def _get_next_worker(self) -> int:
        # todo: calculate and return deadline for batch collection
        # note: Round-Robin + batch-aware + load-balance
        target = (self.current_worker + 1) % self.max_worker
        cnt = 0
        # Round-Robin & load-balance
        with self.control_lock:
            min_qsize = 1e20
            for _ in range(self.max_worker):
                if self.worker_stats[_] == STAT_ON and self.worker_qsize[_] < min_qsize:
                    min_qsize = self.worker_qsize[_]
            # if self.id == 1 and min_qsize > config.LOAD_BALANCE_QSIZE_THRESHOLD:
            #     logging.info('Qsize-{}({}): {}'.format(
            #     self.id, min_qsize, ' '.join(str(_) for _ in self.worker_qsize)))
            while True:
                if self.worker_stats[target] == STAT_ON:
                    if not config.LOAD_BALANCE:
                        break
                    if config.LOAD_BALANCE and self.worker_qsize[target] \
                            <= min_qsize + config.LOAD_BALANCE_QSIZE_THRESHOLD:
                        break
                target = (target + 1) % self.max_worker
                cnt += 1
                if cnt >= self.max_worker:
                    logging.error('Frontend-{}: dispatcher error'.format(self.id))
                    print('# {}'.format(self.worker_qsize))
                    print('# {}'.format(self.worker_stats))
                    print('# min={} {} {}'.format(min_qsize, cnt, target))
                    raise RuntimeError
        self.current_worker = target
        return self.current_worker

    # component
    def dispatcher(self):
        if DISPATCH == 'batching':
            # Round-Robin + batch-aware
            while self.connected:
                wid, size = self._get_next_worker(), self.batch_size
                reply: service_pb2.Reply = None
                for _ in range(size):  # fixme: the last batch maybe not enough
                    try:  # wait for 5s -> empty -> stop...
                        req: service_pb2.Request = self.request_queue.get(timeout=5.0)
                        req.info['batch_size'] = size
                        if _ == 0:
                            req.info['first_in_batch'] = 1  # note: avoid batch splitting
                        reply = self.worker_stubs[wid].SendRequest(req)
                        with self.control_lock:
                            self.worker_qsize[wid] = reply.qsize
                    except queue.Empty:
                        logging.error('Frontend-{}: there\'s no enough requests to generate a batch...'.format(self.id))
                        break
        elif DISPATCH == 'round-robin':
            # Round-Robin
            while self.connected:
                wid = self._get_next_worker()
                try:  # wait for 5s -> empty -> stop...
                    req: service_pb2.Request = self.request_queue.get(timeout=5.0)
                    reply = self.worker_stubs[wid].SendRequest(req)
                    with self.control_lock:
                        self.worker_qsize[wid] = reply.qsize
                except queue.Empty:
                    logging.warning('Frontend-{}: there\'s no requests'.format(self.id))
                    break
        else:
            raise NotImplementedError
        logging.warning('Frontend-{}: dispatcher exited...'.format(self.id))

    # component
    def logger(self):
        stat_que = multiprocessing.Queue()
        _logger = Logger(self.id, os.path.join(self.logdir, 'frontend-{}'.format(self.id)),
                         'frontend-{}'.format(self.id), duration=self.duration, stat_que=stat_que)
        _logger.start()

        deadline = time.time() + LOGGING_INTERVAL * 0.001
        while self.connected:
            qsize, maxq = 0, 0
            for _ in range(self.max_worker):
                if self.worker_stats[_] == STAT_ON:
                    qsize += self.worker_qsize[_]
                    maxq = max(maxq, self.worker_qsize[_])
            stat_que.put({'step': self.step[0], 'batch_size': self.batch_size,
                          'running': self.running_worker, 'booting': self.booting_worker, 'idle': self.idle_worker,
                          'rate_actual': self.rate_current, 'rate_predict': self.rate_future,
                          'sf_actual': self.sf_current, 'sf_predict': self.sf_future,
                          'qsize_sum': qsize, 'qsize_max': maxq})

            logging.debug('Frontend-{} qsize: {}'.format(self.id, self.worker_qsize))
            logging.debug('Frontend-{} run/boot:{}/{} - {}'.format(
                self.id, self.running_worker, self.booting_worker, self.worker_stats))
            assert self.running_worker * 3 + self.idle_worker * 2 + self.booting_worker * 1 == np.sum(self.worker_stats)

            time.sleep(max(0, deadline - time.time()))
            deadline += LOGGING_INTERVAL * 0.001
        _logger.join()


if __name__ == '__main__':
    frontends = []
    for i in range(FRONTEND_NUM):
        frontend = Frontend(
            fid=i,
            ip=FRONTEND_IP[i],
            port=FRONTEND_PORT[i]
        )
        frontend.start()
    for frontend in frontends:
        frontend.join()
