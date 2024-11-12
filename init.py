import sys
import time
from dynamixel_controller import DynamixelController
import constants 

# Constants
baudrate = constants.BAUDRATE
state_dim = constants.STATE_DIM

# サーボIDの設定
follower1_ids = list(range(1, state_dim + 1))  # 1からstate_dimまでのフォロワーID
leader1_ids = list(range(1, state_dim + 1))    # 1からstate_dimまでのリーダーID

# フォロワーサーボの初期ゴールポジション
initial_goal_positions = [2047] * state_dim

# Create controller instances
controller_follower = DynamixelController(constants.FOLLOWER0, baudrate)
controller_leader = DynamixelController(constants.LEADER0, baudrate)

# Attempt to set up the ports
if not controller_follower.setup_port():
    sys.exit("フォロワーのポート設定に失敗しました。プログラムを終了します。")
if not controller_leader.setup_port():
    sys.exit("リーダーのポート設定に失敗しました。プログラムを終了します。")

controller_follower.enable_torque(follower1_ids)
controller_leader.enable_torque(leader1_ids)

# Set initial positions for follower servos
success = controller_follower.sync_write_goal_position(
    follower1_ids,
    initial_goal_positions
)
# Set initial positions for follower servos
success = controller_leader.sync_write_goal_position(
    leader1_ids,
    initial_goal_positions
)
if success:
    print("サーボへの初期値の反映に成功しました。")
else:
    print("サーボへの初期値の反映に失敗しました。")


