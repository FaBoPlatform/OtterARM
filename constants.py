# constants.py

import pathlib

### Task parameters
DATA_DIR = './data'  # データフォルダを指定

# ポートを指定
#LEADER0 = "COM4"
#FOLLOWER0 = "COM7"
LEADER0 = "/dev/ttyUSB0"
FOLLOWER0 = "/dev/ttyUSB1"
LEADER1 = "/dev/ttyUSB2"
FOLLOWER1 = "/dev/ttyUSB3"

BAUDRATE = 1000000

# アームのペア数とアームの関節数
PAIR = 2
STATE_DIM = 6

TASK_CONFIGS = {
    'test1': {
        'dataset_dir': DATA_DIR + '/test1',
        'episode_len': 300,
        'num_episodes': 30,
        'camera_names': ['left','right','front','top'],
        'camera_device_ids': [0,2,8,14],
        'camera_formats': ["YUYV","YUYV","MJPG","MJPG"],
        'camera_fps': [30,30,30,30],
        'width': 320,
        'height': 240,
    },
}

TRAIN_CONFIG = {
    'ckpt_dir': 'checkpoint',
    'policy_class': 'ACT',
    'kl_weight': 10,
    'chunk_size': 100,
    'hidden_dim': 512,
    'batch_size': 8,
    'dim_feedforward': 3200,
    'lr': 1e-5,
    'seed': 0,
    'eval': False,
    'onscreen_render': False,
    'temporal_agg': False,
}
