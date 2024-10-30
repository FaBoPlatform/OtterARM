# constants.py

import pathlib

### Task parameters
DATA_DIR = './data'  # データフォルダを指定

# ポートを指定
LEADER0 = "COM4"
FOLLOWER0 = "COM7"
LEADER1 = "COM3"
FOLLOWER1 = "COM6"

BAUDRATE = 1000000

ARM_DIM = 6

LEADER2 = ''
FOLLOWER2 = ''

TASK_CONFIGS = {
    'test1': {
        'dataset_dir': DATA_DIR + '/test1',
        'episode_len': 200,  # ここでエピソードの長さを指定
        'num_episodes': 50,  # ここでエピソードの長さを指定
        'camera_names': ['front','top','left','right'],
        'camera_device_ids': [2,0,1,3],
        'arm_dim': 6,
        'width': 320,
        'height': 240,
    },
}

### Simulation envs fixed constants
DT = 0.02

PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213
