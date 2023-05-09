import grpc
import time
from math import ceil
import threading
import multiprocessing
from concurrent import futures
import simulation_pb2
import simulation_pb2_grpc


class Client:
    # 初始化 Client
    def __init__(self, frontend_address, frontend_list, backend_list, rate, total=1e9, batch_size=4, is_running=True):
        self.frontend_address = frontend_address
        self.channel = grpc.insecure_channel(frontend_address)
        self.stub = simulation_pb2_grpc.FrontendStub(self.channel)

        self.frontend_list = frontend_list
        self.backend_list = backend_list
        self.rate = rate
        self.total = int(total)
        self.batch_size = batch_size
        self.is_running = is_running

        self.latencies = []
        self.start_adj = time.time()
        self.end_adj = time.time()

    def run(self):
        # 通过 self.is_running 控制两个线程同时停止
        client_send_requests = threading.Thread(target=self.send_requests)
        client_dynamic_rate = threading.Thread(target=self.dynamic_rate)
        client_send_requests.start()
        client_dynamic_rate.start()

    # 向 Frontend 发送请求
    def send_requests(self):
        response_futures = []
        for i in range(self.total):
            if not self.is_running:
                break

            time.sleep(max(1 / self.rate.value - (self.end_adj - self.start_adj), 0))
            self.start_adj = time.time()

            start_time = int(time.time() * 1e6)
            frontend_list.append(start_time)
            request = simulation_pb2.Request(id=i, start_time=start_time)
            # 使用异步API
            response_future = self.stub.SendRequest.future(request)
            response_futures.append(response_future)

            self.end_adj = time.time()

        with threading.Lock():
            print("[Client] all requests were sent")

        self.is_running = False

        # 得到所有 result
        for response_future in response_futures:
            response = response_future.result()
            if response.end_times:
                for i in response.end_times:
                    self.backend_list.append(i)

        # 发送结束信号 -1
        request = simulation_pb2.Request(id=len(self.frontend_list), start_time=-1)
        response = self.stub.SendRequest(request)

        # 记录 latencies
        for i in range(len(self.backend_list)):
            self.latencies.append((self.backend_list[i] - self.frontend_list[i]) / 1e3)
        self.print_statistics()

    # 动态调整 rate
    def dynamic_rate(self):
        for i in range(2):
            for j in range(9):
                time.sleep(10)
                if not self.is_running:
                    return
                self.rate.value -= 10 * (i * 2 - 1)  # 10 -> 100 -> 10
                with multiprocessing.Lock():
                    print("[Client] self.rate.value =", self.rate.value)

        self.is_running = False

    # 打印数据
    def print_statistics(self):
        with threading.Lock():
            print("[Client] request latencies =")
            for i in range(len(self.latencies)):
                print(round(self.latencies[i]), end=" ")
                if (i + 1) % 20 == 0:
                    print()
            if len(self.latencies) % 20 != 0:
                print()
            print("[Client] amount of requests =", len(self.latencies))
        self.latencies.sort()
        avg_latency = sum(self.latencies) / len(self.latencies)
        p99_latency = self.latencies[ceil(0.99 * len(self.latencies)) - 1]
        with threading.Lock():
            print(f"max of latencies: {self.latencies[len(self.latencies) - 1]:.1f} ms")
            print(f"avg of latencies: {avg_latency:.3f} ms")
            print(f"99% quantile of latencies: {p99_latency:.1f} ms")


class Frontend(simulation_pb2_grpc.FrontendServicer):
    # 初始化 Frontend
    def __init__(self, workers, server, rate, duration=0.4, batch_size=4, num_workers=0, status=True):
        self.workers = workers
        self.server = server
        self.rate = rate
        self.duration = duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.status = status

        self.queue = []
        self.times = []

    # receiver and sender
    def SendRequest(self, request, context):
        if request.start_time == -1:
            with threading.Lock():
                print("[Frontend] server is stopped")
            self.status = False

        self.times.append(time.time())

        # 第一次调用 SendRequest 时打开 monitor 线程
        if self.num_workers == 0:
            frontend_monitor = threading.Thread(target=self.monitor)
            frontend_monitor.start()
            self.num_workers += 1

        self.queue.append(request)
        end_times = []
        if len(self.queue) >= self.batch_size:
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            worker = self.select_worker()
            end_times = worker.process_batch(batch)

        return simulation_pb2.Response(id=request.id, end_times=end_times)

    # Frontend 的监测器
    def monitor(self):
        with threading.Lock():
            print("[Frontend] monitor is working")
        rates = []
        while True:
            if not self.status:
                self.num_workers = 0
                self.server.stop(0)
            if len(self.times) >= self.batch_size:
                rate = self.batch_size / (self.times[self.batch_size - 1] - self.times[0])
                rates.append(rate)
                self.times = self.times[self.batch_size:]
            if len(rates) >= 1:
                self.num_workers = max(ceil(self.duration * (sum(rates) / len(rates)) / self.batch_size), 1)  # 调整工作的 worker 数量
                rates = []
            # self.num_workers = max(ceil(self.duration * self.rate.value / self.batch_size), 1)  # 调整工作的 worker 数量
            time.sleep(0.01)

    # 选择 worker
    def select_worker(self):
        for i in range(min(self.num_workers + 2, len(self.workers))):
            if not self.workers[i].is_working:
                return self.workers[i]


class Worker:
    # 初始化 Worker
    def __init__(self, name, is_working=False, duration=0.4):
        self.name = name
        self.is_working = is_working
        self.duration = duration
        self.queue = []  # batches of processing / ed
        self.start_adj = time.time()
        self.end_adj = time.time()

    # 处理 batch
    def process_batch(self, batch):
        self.is_working = True
        time.sleep(max(self.duration - (self.end_adj - self.start_adj), 0))
        self.start_adj = time.time()
        self.queue.append(batch)
        end_time = int(time.time() * 1e6)
        end_times = []
        for _ in range(len(batch)):
            end_times.append(end_time)
        latencies = [end_time - request.start_time for request in batch]
        # self.queue.pop(0)
        with threading.Lock():
            print(f"[{self.name}] latency: {', '.join([f'{l / 1e3:.1f}' for l in latencies])} | Qsize={len(self.queue)}")
        self.is_working = False
        self.end_adj = time.time()
        return end_times  # 返回一组结束时间


def serve(frontend_address, rate):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulation_pb2_grpc.add_FrontendServicer_to_server(Frontend([Worker("Worker-1"), Worker("Worker-2"),
                                                                 Worker("Worker-3"), Worker("Worker-4"),
                                                                 Worker("Worker-5"), Worker("Worker-6"),
                                                                 Worker("Worker-7"), Worker("Worker-8"),
                                                                 Worker("Worker-9"), Worker("Worker-10")], server, rate), server)
    server.add_insecure_port(frontend_address)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    manager_1 = multiprocessing.Manager()
    manager_2 = multiprocessing.Manager()
    frontend_list = manager_1.list()  # 存储请求处理开始的时间
    backend_list = manager_2.list()  # 存储请求处理结束的时间
    rate = multiprocessing.Value('i', 10)

    frontend_address = "localhost:50053"
    multiprocessing.Process(target=serve, args=(frontend_address, rate, )).start()

    time.sleep(1)
    client = Client(frontend_address, frontend_list, backend_list, rate)
    client.run()

