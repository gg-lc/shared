import logging
import multiprocessing
import os
import queue
import random
import threading
import time
from concurrent import futures

import grpc
import setproctitle
import tensorboardX

import config
import utils
from protos import service_pb2, service_pb2_grpc


class WorkerServicer(service_pb2_grpc.InferenceServicer):
    # todo: update to multiprocessing.Queue
    def __init__(self, request_que: queue.Queue, ctl_que: queue.Queue,
                 step_l: list, recv_interval: queue.Queue):
        """
        Universal gRPC servicer of worker.
        Receive requests and control messages from frontend.
        :param request_que: Queue used to store and deliver requests.
        :param ctl_que: Queue used to store and deliver control messages.
        :param step_l: shared list, used to record the number of received requests.
        :param recv_interval: Queue used to store the interval between each request.
        """
        self.request_que = request_que
        self.ctl_que = ctl_que
        self.recv_interval = recv_interval
        self.step_l = step_l
        self.last_recv = None  # used to record the interval between each request

    def SendRequest(self, request: service_pb2.Request, context):
        if request.type == service_pb2.TYPE_REQUEST:  # request
            request.latency.append(utils.get_latency_ms(request.begin))
            self.request_que.put(request)
            self.step_l[0] += 1
            if self.last_recv is not None:
                self.recv_interval.put(time.time() - self.last_recv)
            self.last_recv = time.time()
            logging.debug('servicer received request {}({})'.format(
                request.rid, request.sub
            ))
            # logging.debug('servicer received request: {}'.format(request))
        else:
            self.ctl_que.put(request)
            logging.debug('servicer received a control msg. type={}'.format(
                request.type
            ))
            # logging.debug('servicer received a control msg: {}'.format(request))
        return service_pb2.Reply(qsize=self.request_que.qsize())


class Worker(multiprocessing.Process):
    def __init__(self, ip: str, port: str or int, processes: int):
        """
        Worker:
            worker = Worker()
            worker.start()
            worker.join()
        :param ip: Ip address
        :param port: gRPC listening port
        :param processes: (to be done) number of parallel processes to infer requests
        """
        # initialization
        super().__init__()
        self.id = None  # int
        self.port = str(port)
        self.addr = ip + ':' + str(port)
        self.logdir = None
        utils.set_logging_format()

        # status
        self.running_lock = None  # used to control the running of processor (proc.get_lock -> run)
        self.connected = False
        self.stat = None  # fixme: the processor gathers batch only when stat = ON|IDLE

        # receiver & queue
        self.server = None
        self.step_l = [0]  # receive a request -> +1
        self.done_tmp = 0  # number of finished requests in a period of time
        self.done_lat = .0  # total latency of finished requests in a period of time
        self.drop_tmp = 0  # number of dropped requests in a period of time
        self.ctl_que = queue.Queue()
        # Servicer -> [req_que] -> Processor -> [batch_que] -> Inference
        # -> [res_que] -> Post_processor -> [send_que] -> Sender
        self.req_que = queue.Queue()
        self.batch_que = queue.Queue()
        self.res_que = queue.Queue()
        self.send_que = queue.Queue()  # multiprocessing.Queue()
        self.recv_intervals = queue.Queue()  # used to monitor the receiving rate
        self.send_intervals = queue.Queue()  # used to monitor the process rate

        # processor
        self.duration = None
        self.processes = processes
        self.model_name = None
        self.model = None
        self.batch_size = None
        self.slo = None
        self.module_slo = None  # slo * n (frontend-n)

        # policy
        self.dummy_request = None
        self.dummy_img = None
        self.control_lock = threading.Lock()
        self.timer_shutdown: threading.Timer = None

        # sender
        self.addr_client = None
        self.addr_frontend = None
        self.addr_next = None

    def run(self) -> None:
        setproctitle.setproctitle('Worker-{}'.format(self.id))

        # todo: (1) tf-serving
        # start tf/s
        # -> self.model_path...
        # -> self.model_name...
        # -> self.serving_port...
        # ->
        # os.system(...) -> load init model
        ...
        # create channel
        # self.serving_infer_stub = ...
        # self.serving_ctl_stub = ...
        ...

        # create gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = WorkerServicer(self.req_que, self.ctl_que, self.step_l, self.recv_intervals)
        service_pb2_grpc.add_InferenceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port('[::]:' + self.port)
        self.server.start()
        logging.info('Worker-{} start listening at {}'.format(self.id, self.addr))

        # create controller, processor, sender and monitor
        _controller = threading.Thread(target=self.controller)
        _pre_processor = threading.Thread(target=self.pre_processor)
        # _post_processor = threading.Thread(target=self.post_processor)
        _sender = threading.Thread(target=self.sender)
        _monitor = threading.Thread(target=self.monitor)

        # initialize
        msg = self.ctl_que.get()  # wait for connection (type=CONNECT)
        logging.debug('Worker received the connection request:\n {} {}'.format(msg.info, msg.msg))
        # uint32 info
        self.id = msg.info['id']
        self.batch_size = msg.info['batch_size']
        self.stat = msg.info['stat']
        self.slo = msg.info['slo']
        self.module_slo = msg.info['module_slo']
        # string info
        self.model_name = msg.msg['model_name']
        self.logdir = msg.msg['logdir']
        self.duration = utils.load_profile(self.model_name, config.AOT_TUNER)['duration']

        # dummy request
        self.dummy_request = service_pb2.Request(
            type=service_pb2.TYPE_DUMMY,
            bytes=msg.bytes
        )
        self.dummy_img = utils.bytes2img(msg.bytes)

        # todo: move client addr into each request -> support multi-client
        self.addr_client = msg.msg['client_addr']
        self.addr_frontend = msg.msg['frontend_addr']
        self.addr_next = msg.msg['next_frontend'] if 'next_frontend' in msg.msg.keys() else None
        logging.info('Worker-{} start running. model:{} slo:{} batch_size:{} stat:{}'.format(
            self.id, self.model_name, self.slo, self.batch_size, self.stat
        ))

        self.connected = True
        self.running_lock = threading.Lock()

        setproctitle.setproctitle('Worker-{}'.format(self.id))

        _controller.start()
        _monitor.start()
        _pre_processor.start()
        # _post_processor.start()
        _sender.start()

        try:
            while self.connected:
                time.sleep(3)
        except KeyboardInterrupt:
            logging.warning('Worker-{} caught KeyboardInterrupt, exiting...'.format(self.id))
            self.connected = False
        finally:
            logging.warning('Worker-{} disconnected, trying to exit...'.format(self.id))
            _controller.join()
            _monitor.join()
            _pre_processor.join()
            # _post_processor.join()
            _sender.join()

            # todo: (2) close tf-serving
            ...

        logging.warning('Worker-{} exited'.format(self.id))

    def monitor(self):  # log & heartbeat
        writer = None
        if config.WORKER_LOG:
            writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(self.logdir, 'worker-{}'.format(self.id)),
                flush_secs=3,
                # fixme: need to distinguish the workers w/ a same id from different Frontend...
                #   add frontend id or use global worker id?
                filename_suffix='.worker-{}'.format(self.id),
            )
            logging.info('Worker-{}.monitor started. logdir: {}'.format(self.id, self.logdir))
        else:
            logging.info('Worker-{}.monitor started WITHOUT logger...'.format(self.id))

        deadline = time.time()
        while self.connected:
            # snapshot
            step, stat, batch_size = self.step_l[0], self.stat, self.batch_size
            finished, dropped, total_latency = self.done_tmp, self.drop_tmp, self.done_lat
            finish_rate = finished / (finished + dropped) if (finished + dropped) > 0 else 1.0
            avg_latency = total_latency / finished if finished else .0
            self.done_tmp = 0
            self.done_lat = 0
            self.drop_tmp = 0
            que_size = self.req_que.qsize()

            # calculate recv rate & send rate
            recv_num, send_num = self.recv_intervals.qsize(), self.send_intervals.qsize()
            total_t = .0
            for _ in range(recv_num):
                total_t += self.recv_intervals.get()
            recv_rate = recv_num / total_t if total_t else .0
            total_t = .0
            for _ in range(send_num):
                total_t += self.send_intervals.get()
            send_rate = send_num / total_t if total_t else .0

            logging.debug('Worker-{}: Qsize={} BS={} avg_lat={:.2}ms'.format(
                self.id, self.req_que.qsize(), self.batch_size, avg_latency
            ))

            deadline = deadline + config.HEARTBEAT_INTERVAL * 0.001
            # heartbeat
            heartbeat = service_pb2.Request(type=service_pb2.TYPE_HEARTBEAT)
            heartbeat.info['id'] = self.id
            heartbeat.info['stat'] = stat
            heartbeat.info['qsize'] = que_size
            heartbeat.info['avg_lat'] = int(avg_latency)
            heartbeat.info['proc_rate'] = int(send_rate)
            heartbeat.info['batch_size'] = batch_size
            self.send_que.put(heartbeat)
            # logging
            if writer is not None:
                writer.add_scalar('worker-{}xx/stat'.format(self.id // 100), stat, step)
                writer.add_scalar('worker-{}xx/avg_latency'.format(self.id // 100), avg_latency, step)
                writer.add_scalar('worker-{}xx/batch_size'.format(self.id // 100), batch_size, step)
                writer.add_scalar('worker-{}xx/queue_size'.format(self.id // 100), que_size, step)
                writer.add_scalar('worker-{}xx/finish_rate'.format(self.id // 100), finish_rate, step)
                # writer.add_scalar('worker-{}xx/recv_rate'.format(self.id // 100), recv_rate, step)
                # writer.add_scalar('worker-{}xx/send_rate'.format(self.id // 100), send_rate, step)
                # writer.add_scalar('worker-{}xx/capability'.format(self.id // 100),
                #                   self.batch_size * 1000 / self.duration[self.batch_size], step)

            time.sleep(max(0, deadline - time.time()))
        if writer is not None:
            writer.close()
        logging.warning('Worker-{} monitor exited...'.format(self.id))

    def update_stat(self, new):
        def _shutdown():
            with self.control_lock:
                _pre = self.stat
                self.stat = config.STAT_OFF
                self.send_que.put(service_pb2.Request(
                    type=service_pb2.TYPE_CONTROL, info={'id': self.id % 100, 'stat': config.STAT_OFF}))
            logging.info('worker-{} shutdown... (stat: {}->{})'.format(self.id, _pre, self.stat))

        # off boot [idle] on
        logging.info('Worker-{} try to update stat: {}->{}'.format(self.id, self.stat, new))
        with self.control_lock:
            pre = self.stat
            if pre == new:
                ...
            # shutdown current worker
            elif new == config.STAT_OFF:  # *->off
                if self.stat == config.STAT_ON:  # on->off | on->idle->off(timer)
                    if config.TURN_OFF_DELAY:  # turn of delay
                        self.stat = config.STAT_IDLE
                        self.timer_shutdown = threading.Timer(
                            config.DELAYED_OFF_TIME * 0.001, function=_shutdown)
                        self.timer_shutdown.start()
                        logging.info('worker-{} will shutdown after {} ms'.format(self.id, config.DELAYED_OFF_TIME))
                    else:
                        self.stat = config.STAT_OFF
                        self.send_que.put(service_pb2.Request(
                            type=service_pb2.TYPE_CONTROL, info={'id': self.id % 100, 'stat': config.STAT_OFF}))
                        logging.info('worker-{} shutdown...'.format(self.id))
                elif self.stat == config.STAT_IDLE:  # idle -> off (call timer at once)
                    logging.warning('worker-{} is already in IDLE...'.format(self.id))
                else:  # [config.STAT_OFF, config.STAT_BOOT]:
                    logging.error('Worker-{} state update error'.format(self.id))

            # launch current worker
            elif new == config.STAT_ON:
                if self.stat == config.STAT_IDLE:  # quick start
                    self.timer_shutdown.cancel()
                    self.stat = config.STAT_ON
                    self.send_que.put(service_pb2.Request(
                        type=service_pb2.TYPE_CONTROL,
                        info={'id': self.id % 100, 'stat': config.STAT_ON}))
                    logging.info('Worker-{} quick start: IDLE -> ON'.format(self.id))
                elif self.stat == config.STAT_OFF:
                    self.stat = config.STAT_BOOT
                    time.sleep(config.WORKER_BOOT_TIME * 0.001)
                    self.stat = config.STAT_ON
                    self.send_que.put(service_pb2.Request(
                        type=service_pb2.TYPE_CONTROL,
                        info={'id': self.id % 100, 'stat': config.STAT_ON}
                    ))
                    logging.info('Worker-{} normal start: OFF -> ON'.format(self.id))
                else:  # [config.STAT_ON, config.STAT_BOOT]:
                    logging.error('Worker-{} state update error'.format(self.id))

            # warmup current worker
            elif new == config.STAT_IDLE:
                if self.stat == config.STAT_OFF:
                    logging.info('worker-{} received warmup signal. booting...')
                    self.stat = config.STAT_BOOT
                    time.sleep(config.WORKER_BOOT_TIME * 0.001)
                    self.stat = config.STAT_IDLE
                    if self.timer_shutdown is not None:
                        self.timer_shutdown.cancel()
                    self.timer_shutdown = threading.Timer(
                        config.DELAYED_OFF_TIME * 0.001, function=_shutdown)
                    self.timer_shutdown.start()
                    logging.info('worker-{} warmed up. Shutdown after {} ms'.format(self.id, config.DELAYED_OFF_TIME))
                else:
                    logging.error('Worker-{} stat update error {}->{}'.format(self.id, self.stat, new))

            else:
                logging.warning('Invalid stat: {}'.format(new))
            logging.info('Worker-{} updated stat: {}->{}'.format(self.id, pre, self.stat))
            # logging.warning('Worker-{} updated stat: {}->{}'.format(self.id, pre_stat, new_stat))

    def controller(self):  # controller
        while self.connected or not self.ctl_que.empty():
            msg = self.ctl_que.get()  # type=CONTROL|CONNECT
            logging.info('worker-{} received control msg (type={})'.format(self.id, msg.type))
            logging.debug('worker-{} received control msg:\n {}'.format(self.id, msg))

            # disconnect msg
            if msg.type == service_pb2.TYPE_CONNECT:
                if msg.info['msg'] == 0:
                    self.connected = False
                    logging.warning('worker-{} controller exited...'.format(self.id))
                    break
                else:
                    logging.warning('worker-{}: unknown connection msg:\n{}'.format(self.id, msg))
            # update stat | batch_size
            elif msg.type == service_pb2.TYPE_CONTROL:
                for k in msg.info.keys():
                    # todo: update batch_size should always before updating stat
                    if k == 'batch_size':
                        logging.info('Worker-{} batch_size: {}->{}'.format(self.id, self.batch_size, msg.info[k]))
                        self.batch_size = msg.info[k]
                    elif k == 'stat':
                        logging.info('Worker-{} try to update stat: {}->{}'.format(self.id, self.stat, msg.info[k]))
                        self.update_stat(msg.info[k])
                    else:
                        logging.error('worker-{}: unsupported control key: {}'.format(self.id, k))
            # unknown control message type
            else:
                logging.warning('worker-{}: unknown msg type: {}'.format(self.id, msg))

    def get_batch(self, batch_size: int = None, wait_time: float = None) \
            -> (list, list) or (None, None):
        # fixme: add dummy request into the last batch if there's no enough requests...
        """
        Get a batch of requests from req_queue
        :param batch_size: int, default is None.
            None: get a whole batch received from Frontend (NO drop, NO dummy)
            int: get a batch of requests that will not violate the SLO [drop | dummy]
        :param wait_time: float, return (None, None) if there's no enough requests to
            generate a batch (the last batch...)
        :return: (requests, batch)
        """

        def get_request(_timeout=None):
            _req = self.req_que.get(timeout=_timeout)
            self.req_que.task_done()
            return _req

        def is_timeout(_req: service_pb2.Request = None, _batch_size=batch_size):
            if req is None:
                return True
            return utils.get_latency_ms(_req.begin) + self.duration[_batch_size] > self.module_slo

        def drop_req(_req: service_pb2.Request):  # tag drop & send back to client
            self.drop_tmp += 1
            _req.type = service_pb2.TYPE_RESULT
            _req.bytes = b''
            _req.info['done'] = 0
            self.send_que.put(_req)
            # logging.warning('Worker-{} dropped request {}({})'.format(self.id, _req.rid, _req.sub))

        # round-robin dispatch & early drop / lazy drop
        if config.DISPATCH == 'round-robin':
            size = self.batch_size
            requests, batch = [], []
            if config.DROP == 'none':
                try:
                    for _ in range(size):
                        req = get_request(_timeout=wait_time)
                        batch.append(utils.bytes2img(req.bytes))
                        requests.append(req)
                    return requests, batch
                except queue.Empty:
                    logging.warning('Worker-{} request is not enough...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None
            elif config.DROP == 'early':
                try:
                    for _ in range(size):
                        req = get_request(_timeout=wait_time)
                        batch.append(utils.bytes2img(req.bytes))
                        requests.append(req)
                    for req in requests:
                        print('worker-{} req-{}/b{} lat dur slo: {} {} {}'.format(
                            self.id, req.rid, self.batch_size,
                            utils.get_latency_ms(req.begin),
                            self.duration[size], self.module_slo
                        ))
                    while is_timeout(requests[0], size):
                        drop_req(requests[0])
                        logging.warning('Worker-{} early drop req-{}'.format(self.id, requests[0].rid))
                        requests.pop(0)
                        batch.pop(0)
                        req = get_request(_timeout=wait_time)
                        batch.append(utils.bytes2img(req.bytes))
                        requests.append(req)
                    return requests, batch
                except queue.Empty:
                    logging.warning('Worker-{} request is not enough...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None
            elif config.DROP == 'lazy':
                try:
                    for _ in range(size):
                        req = get_request(_timeout=wait_time)
                        if is_timeout(req, size):
                            drop_req(req)
                            logging.warning('Worker-{} lazy drop req-{}'.format(self.id, req.rid))
                            continue
                        batch.append(utils.bytes2img(req.bytes))
                        requests.append(req)
                    if len(requests) == 0:
                        return None, None
                    return requests, batch
                except queue.Empty:
                    logging.warning('Worker-{} request is not enough...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None
            else:
                raise NotImplementedError

        # batching-dispatch
        if batch_size is None:
            requests, batch = [], []
            size, timeouts = None, 0
            while True:  # get the first request of a batch
                try:
                    req = get_request(_timeout=wait_time)
                    if req.info['first_in_batch'] == 1:
                        req.info['first_in_batch'] = 0
                        batch.append(utils.bytes2img(req.bytes))
                        requests.append(req)
                        size = req.info['batch_size']
                        timeouts += is_timeout(req, size)
                        break
                    else:
                        logging.error('Worker-{} error in get batch, drop: {}'.format(self.id, req.rid))
                        drop_req(req)
                except queue.Empty:
                    logging.warning('Worker-{}: No request in the queue'.format(self.id))
                    return None, None

            # get batch
            for _ in range(size - 1):
                try:
                    req = get_request(_timeout=wait_time)
                    requests.append(req)
                    batch.append(utils.bytes2img(req.bytes))
                    timeouts += is_timeout(req, size)
                except queue.Empty:
                    logging.error('Worker-{} error in get batch, drop batch...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None

            if config.DROP == 'adaptive':
                if timeouts >= size // 2:
                    # for req in requests:
                    #     print('worker-{} req-{}/b{} lat dur slo: {} {} {}'.format(
                    #         self.id, req.rid, self.batch_size,
                    #         utils.get_latency_ms(req.begin),
                    #         self.duration[size], self.module_slo
                    #     ))
                    logging.warning('Worker-{} Adaptive Batch Drop drop a batch...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None
                logging.info('Worker-{} get a batch, size={}({})'.format(self.id, size, self.batch_size))
                return requests, batch
            elif config.DROP == 'queue':
                if self.req_que.qsize() >= size:
                    logging.warning('Worker-{} Queue Drop drop a batch...'.format(self.id))
                    for req in requests:
                        drop_req(req)
                    return None, None
                logging.info('Worker-{} get a batch, size={}({})'.format(self.id, size, self.batch_size))
                return requests, batch
            else:
                logging.error('Worker-{} unsupported drop policy: {}'.format(self.id, config.DROP))
                raise NotImplementedError

            # try:  # try to get a batch from req_que
            #     # get the first request and batch size
            #     req = get_request(_timeout=wait_time)  # fixme: catch queue.Empty
            #     requests.append(req)
            #     batch.append(utils.bytes2img(req.bytes))
            #     # get a batch
            #     num_timeout = 0
            #     size = req.info['batch_size']
            #     for _ in range(size - 1):
            #         req = get_request(_timeout=wait_time)
            #         num_timeout += is_timeout(req, size)
            #         requests.append(req)
            #         batch.append(utils.bytes2img(req.bytes))
            #     assert requests[-1].info['last_in_batch'] == 1  # avoid batch splitting -> frontend.dispatcher
            #     requests[-1].info['last_in_batch'] = 0
            # except queue.Empty:
            #     for req in requests:
            #         logging.warning('Worker-{} dropped the last batch: rid={}'.format(self.id, req.rid))
            #     return None, None
            # # drop batch if:
            # #   1. config.DROP
            # #   2. there's another batch in the queue
            # #   3. current batch has timeout requests
            # if config.DROP and self.req_que.qsize() >= size and num_timeout:
            #     for req in requests:
            #         drop_req(req)
            #     requests, batch = self.get_batch(batch_size, wait_time)  # recursion, get a new batch
            # if requests is not None:
            #     logging.debug('Worker-{} get a batch {}/{}'.format(self.id, len(requests), len(batch)))
            # return requests, batch

        # round-robin dispatch
        else:
            raise NotImplementedError

        # note: The following function is deprecated
        # else:
        #     logging.error('@deprecated: current get batch method is deprecated...')
        #     # todo: this function costs a long time to preprocess the request
        #     #   -> requires multiple threads the execute in parallel
        #     dummy, drop = config.DUMMY, config.DROP
        #     requests, batch = [], []
        #
        #     # step-1: get requests
        #     if not dummy and not drop:  # worst: not dummy & no drop
        #         for _ in range(batch_size):
        #             requests.append(get_request())
        #             batch.append(utils.bytes2img(requests[-1].bytes))
        #
        #     elif not dummy and drop:
        #         # drop request needs dummy request
        #         raise AttributeError
        #
        #     elif dummy and not drop:  # add dummy requests when timeout
        #         requests.append(get_request())
        #         batch.append(utils.bytes2img(requests[-1].bytes))
        #
        #         deadline = utils.timestamp2float(requests[0].begin) + self.module_slo * 0.001
        #         for _ in range(batch_size - 1):
        #             try:
        #                 request = get_request(_timeout=max(0, deadline - time.time()))
        #                 requests.append(request)
        #                 batch.append(utils.bytes2img(request.bytes))
        #             except queue.Empty:
        #                 break
        #
        #         if len(requests) < batch_size:
        #             logging.warning('Add {} dummy requests into a batch {}/{}'.format(
        #                 batch_size - len(requests), len(requests), batch_size
        #             ))
        #             while len(requests) < batch_size:
        #                 requests.append(self.dummy_request)
        #                 batch.append(self.dummy_img)
        #
        #     else:  # dummy & drop: best
        #         # todo: recheck the logic here / verify the function
        #         # deadline = request.begin + slo
        #         # check request: deadline - duration >= current
        #         # wait deadline: min{deadline} - duration
        #         # wait timeout: max(0, wait deadline - current)
        #         min_ddl, wait_timeout = 1e12, None
        #         for _ in range(batch_size):
        #             try:
        #                 # request = None
        #                 while True:
        #                     request = get_request(_timeout=wait_timeout)
        #                     if is_timeout(request):
        #                         drop_req(request)
        #                     else:
        #                         break
        #                 # add request
        #                 request.latency.append(utils.get_latency_ms(request.begin))
        #                 requests.append(request)
        #                 batch.append(utils.bytes2img(request.bytes))
        #                 # update time
        #                 min_ddl = min(min_ddl,
        #                               utils.timestamp2float(request.begin) + (
        #                                       self.module_slo - self.duration[batch_size]) * 0.001)
        #                 wait_timeout = max(0, min_ddl - time.time())
        #             except queue.Empty:  # dummy
        #                 break
        #
        #         if len(requests) < batch_size:
        #             logging.warning('Add {} dummy requests into a batch {}/{}'.format(
        #                 batch_size - len(requests), len(requests), batch_size
        #             ))
        #             while len(requests) < batch_size:
        #                 requests.append(self.dummy_request)
        #                 batch.append(self.dummy_img)
        #
        #     logging.debug('Worker-{} get a batch {}/{}'.format(self.id, len(requests), len(batch)))
        #     return requests, batch

    def pre_processor(self):
        """
        Gather batches into the batch_que based on the
        requirements of the inference processes (Semaphore).
        Determine whether to drop the request or add dummy
        requests based on the configuration in config.py.
        """
        # todo: load model
        _semaphore = threading.Semaphore(self.processes)
        _infers, _events = [], []

        for _ in range(self.processes):
            event = threading.Event()
            _infers.append(threading.Thread(
                target=self.inference,
                args=(self.batch_que, self.send_que, _semaphore, event, self.model_name, self.id)
            ))
            _events.append(event)
            _infers[_].start()

        # todo: optimize request drop policy -> maximum reward
        while self.connected:
            # start gathering a batch if any inference process has just started an inference
            # purpose: avoid batch waiting at batch_que
            _semaphore.acquire()
            batch_size = self.batch_size
            # requests, batch = self.get_batch(batch_size)  # todo: dummy & drop needs batch_size
            try:
                requests, batch = self.get_batch(wait_time=5)  # fixme: batch_size
                if requests is None:
                    _semaphore.release()
                    continue
                # clear the original request content
                for _ in range(len(requests)):
                    requests[_].latency.append(utils.get_latency_ms(requests[_].begin))
                self.batch_que.put((requests, batch))
            except queue.Empty:
                logging.warning('Worker-{}: There\'s no enough requests.'.format(self.id))
                _semaphore.release()
                continue

        for e in _events:  # stop inference
            e.set()
        logging.warning('Worker-{} wait the inference to stop...'.format(self.id))
        for _ in _infers:
            _.join()
        logging.warning('Worker-{}: processor exited'.format(self.id))

    @staticmethod
    def inference(
            batch_que: queue.Queue,
            res_que: queue.Queue,
            semaphore: threading.Semaphore,
            stopped: threading.Event,
            model: str,
            wid: int
    ):
        profile = utils.load_profile(model, config.AOT_TUNER)
        duration = profile['duration']
        first = profile['first']
        is_batch_size_tuned = [0] * 1024

        # model = load_model(model_name)
        tensor = None  # avoid memory leaks

        while not stopped.is_set():
            # get a batch from batch queue
            try:
                requests, batch = batch_que.get(timeout=5)
            except queue.Empty:
                continue
            semaphore.release()  # subscribe
            logging.debug('Start inferring a batch')
            size = len(batch)
            # tensor = tf.convert_to_tensor(np.array(batch))  # 1~5 * batch_size  (~10-30ms)

            # todo: (3) inference
            # batch = batch_que.get(...)
            # res = self.serving_stub.Predict(...)
            # res_que.put(res)

            # inference
            if is_batch_size_tuned[size] == 1:
                time.sleep(duration[size] * 0.001)
            else:
                logging.warning('Worker-{} bs-{} Autotune={}ms'.format(wid, size, first[size]))
                is_batch_size_tuned[size] = 1
                time.sleep(first[size] * 0.001)

            # postprocess: request -> result | sub-request
            for req in requests:
                req.latency.append(utils.get_latency_ms(req.begin))
                # create sub-request
                if req.type == service_pb2.TYPE_DUMMY:
                    continue
                elif len(req.sub) != 0:  # sub-request
                    logging.debug('Worker-{} infer batch, send to next module'.format(wid))
                    req: service_pb2.Request = req
                    for _ in range(req.sub[0]):
                        sub = service_pb2.Request()
                        sub.CopyFrom(req)
                        sub.sub.pop(0)
                        res_que.put(sub)
                # convert the request to result
                else:
                    logging.debug('Worker-{} infer batch, send to worker'.format(wid))
                    req.info['res'] = random.randint(0, 100)
                    req.info['done'] = 1
                    req.type = service_pb2.TYPE_RESULT
                    res_que.put(req)
            logging.info('Worker-{} finished inferring a batch.'.format(wid))

    def post_processor(self):
        while self.connected or not self.res_que.empty():
            try:
                req, res = self.res_que.get(timeout=5)
            except queue.Empty:
                continue
            # logging

    def sender(self):
        # fixme: create separate stubs for request, results and message
        #   in different function (with ... as stub: ...) -> multi-thread
        # connect to client, frontend [and next frontend if needed]
        client_channel = grpc.insecure_channel(self.addr_client)
        client_stub = service_pb2_grpc.InferenceStub(client_channel)
        frontend_channel = grpc.insecure_channel(self.addr_frontend)
        frontend_stub = service_pb2_grpc.InferenceStub(frontend_channel)
        if self.addr_next is None or self.addr_next == '':
            next_stub = None
        else:
            next_channel = grpc.insecure_channel(self.addr_next)
            next_stub = service_pb2_grpc.InferenceStub(next_channel)

        last_send_req = -1
        while self.connected or not self.send_que.empty():
            try:
                request = self.send_que.get(timeout=5)
            except queue.Empty:
                continue
            # logging
            if request.type in [service_pb2.TYPE_REQUEST, service_pb2.TYPE_RESULT]:
                # record send intervals
                if last_send_req != -1:
                    self.send_intervals.put(time.time() - last_send_req)
                last_send_req = time.time()
                # record number of finished request
                self.done_tmp += 1
                self.done_lat += utils.get_latency_ms(request.begin)

            # done: send back to client
            if request.type == service_pb2.TYPE_RESULT:
                request.bytes = b''
                client_stub.SendRequest(request)
                logging.debug('Worker-{} send back a result {}({})'.format(
                    self.id, request.rid, request.sub
                ))
            # sub-request: send to next frontend
            elif request.type == service_pb2.TYPE_REQUEST:
                if next_stub is None:
                    raise grpc.RpcError
                next_stub.SendRequest(request)
                logging.debug('Worker-{} send sub-request {}({}) to next frontend'.format(
                    self.id, request.rid, request.sub
                ))
            # heartbeat: send to current frontend
            # launch msg: send to current frontend
            elif request.type in [service_pb2.TYPE_HEARTBEAT, service_pb2.TYPE_CONTROL]:
                frontend_stub.SendRequest(request)
            else:
                logging.error('Worker-{}: unknown send msg type:\n {}'.format(self.id, request))
        logging.warning('Worker-{}: sender exited...'.format(self.id))


if __name__ == '__main__':
    workers = []
    for f in range(len(config.WORKER_NUM)):
        for i in range(config.WORKER_NUM[f]):
            workers.append(Worker(
                ip=config.WORKER_IP[f][i],
                port=config.WORKER_PORT[f][i],
                processes=config.WORKER_PROCESSES[f]
            ))
            workers[-1].start()
    for worker in workers:
        worker.join()
