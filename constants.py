# constants.py

import pathlib

### Task parameters
DATA_DIR = './data'  # データフォルダを指定

# ポートを指定
#LEADER0 = "COM4"
#FOLLOWER0 = "COM7"
LEADER0 = "/dev/tty.usbserial-FT94EMKQ"
FOLLOWER0 = "/dev/tty.usbserial-FT94EMR8"
LEADER1 = "COM3"
FOLLOWER1 = "COM6"

BAUDRATE = 1000000

# アームのペア数とアームの関節数
PAIR = 1
STATE_DIM = 5

TASK_CONFIGS = {
    'test1': {
        'dataset_dir': DATA_DIR + '/test1',
        'episode_len': 200,
        'num_episodes': 30,
        'camera_names': ['front'],
        'camera_device_ids': [0],
        'camera_port': [0],
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
