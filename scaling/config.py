""" ===================== part-0: setup  ======================"""
# setup
LOGDIR = '0509-test'
LOGNAME = 'SLOpt'
PORT_OFFSET = 10000
WORKER_LOG = True
LOGGING_LEVEL = 'INFO'  # DEBUG INFO WARNING ERROR
LOGGING_INTERVAL = 1000  # ms, client & frontend
# tensorboard filter example: wiki.*cascade.*(client|frontend|worker)


""" ===================== part-1: workload ====================== """
# workload
WORKLOAD = ['wiki', '2']  # (wiki|tweet|test)-(m30_10r|int[0-9])
FLUCTUATION_RANGE = 0.5  # ± x ms -> ~95% flu within this range
CHANGE_GAP = 200  # change fluctuation per xxx req
RAND_SEED = 101

# WORKLOAD_SLICE = [0, 0.2]  # twitter
# INIT_QPS = [300, 600]
# INIT_SF = [2.0, 0]
# OVERHEAD = 40

WORKLOAD_SLICE = [0.65, 1]  # wiki
INIT_QPS = [170, 340]
INIT_SF = [2.0, 0]
OVERHEAD = 20
GAP_FACTOR = 1

# app-fixed
# MODEL = ['fixed', 'fixed']
# MODULE_SLO = [300, 600]  # NOTE: slo for module-2 = 800-400 = 400ms

# app-game
# MODEL = ['centernet_resnet50_v1_fpn_512x512_coco17_tpu-8', 'NASNetLarge']
# MODULE_SLO = [200, 400]

# app-traffic
MODEL = ['sample', 'sample']
MODULE_SLO = [300, 600]  # NOTE: slo for module-2 = 800-400 = 400ms

# scheduler & worker
SCHEDULE_INTERVAL = 6000
TRIGGER_SCHEDULING = True
HEARTBEAT_INTERVAL = 1000  # ms, logging interval & heartbeat interval

# cost
# Nexus & InferLine: idle cost = 1
# SLOpt: idle cost = 0.3
NORMAL_COST = 1 * 0.001  # per second
IDLE_COST = 0.3 * 0.001  # per second

""" ===================== part-2: policy ====================== """
# estimation
# note: reactive for Nexus
# note: InferLine & Nexus: idle cost = 1
PREDICT = True  # high priority -> reactive | predictive
USE_LSTM = True  # low priority
# interval & cold start
WORKER_BOOT_TIME = 3000  # ms, setup time
INTERVAL_ESTIMATION = 1000  # todo: trigger scheduling -> future / current exceed capability -> schedule
# only effective when predict & not use_lstm
# note: 'envelope' for InferLine
# note: InferLine & Nexus: idle cost = 1
ESTIMATION_POLICY = 'PolyReg-2'  # 'LR' | 'PolyReg-2' | 'envelope'
HISTORY_LENGTH = 50 * 1000  # historical data time range used for estimating future (boot_time) QPS

# extra capacity
# note: extra capacity should increase the overhead to limit batch size... (RR)
EXTRA_CAPACITY = .0  # todo: dynamic extra capacity

# fluctuation history
FLUCTUATION = True
PERCENTILE = 0.9  # >= 0.1

# cascade scaling
CASCADE_SCALING = True
CASCADE_EFFECTIVE_DURATION = 5000  # effective time of scaling factor [>=layers*(boot_time+max_duration)]

# state manager
PRE_WARMUP = True  # fixme: predict + cascade warmup
WARMUP_LSTM = True  # fixme: unfinished. if true -> lstm_l predict else warm ratio
WARMUP_RATIO = 0.1  # num_warmup = ceil(num_target * ratio)
TURN_OFF_DELAY = True
DELAYED_OFF_TIME = 8000

# AoT Tuner
AOT_TUNER = True

# dispatch & drop
DISPATCH = 'batching'  # 'batching' | 'round-robin'
# note: you should optimize other policy before using adaptive
#       otherwise the drop rate would be high
# note: you can use the queue drop at any time...
DROP = 'adaptive'  # none | early | lazy | adaptive | queue

# useless
LOAD_BALANCE = False  # useless  # fixme: worker reply qsize -> RR + skip large queue
LOAD_BALANCE_QSIZE_THRESHOLD = 8  # note: >= batch size
# note: load_balance works well w/o drop...
DUMMY = False  # useless

""" ===================== part-3: cluster ====================== """
# client
CLIENT_IP = 'localhost'
CLIENT_PORT = PORT_OFFSET + 1  # 10001

# frontend
FRONTEND_NUM = 2
FRONTEND_IP = ['localhost'] * 2
FRONTEND_PORT = [PORT_OFFSET + 1000, PORT_OFFSET + 1001]  # 11000, 11001

# worker
WORKER_NUM = [20, 40]
WORKER_NUM_MIN = [1, 1]  # minimum worker number
WORKER_PROCESSES = [1, 1]  # cautious: use multi-processes may cause wrong capability calculation
WORKER_IP = [['localhost'] * 100, ['localhost'] * 100]
WORKER_PORT = [list(range(PORT_OFFSET + 2000, PORT_OFFSET + 2100)),
               list(range(PORT_OFFSET + 2100, PORT_OFFSET + 2200))]

""" ==================== part-4: macro ==================== """
# worker status
STAT_OFF = 0
STAT_BOOT = 1
STAT_IDLE = 2
STAT_ON = 3

# message type
REQUEST_TYPE_NAME = {
    0: 'UNKNOWN',
    1: 'CONNECT',
    2: 'REQUEST',
    3: 'RESULT',
    4: 'CONTROL',
    5: 'MESSAGE',
    6: 'HEARTBEAT',
    7: 'DUMMY',
    8: 'CASCADE'
}
