import sys
import time
from dynamixel_controller import DynamixelController
import constants 

# Constants
BAUDRATE = 1000000

# アドレス定義
ADDR_XL_GOAL_POSITION = 116  # 使用しているモデルに応じて変更

arm_dim = constants.ARM_DIM

# サーボIDの設定
follower1_ids = list(range(1, arm_dim + 1))  # 1からarm_dimまでのフォロワーID
leader1_ids = list(range(1, arm_dim + 1))    # 1からarm_dimまでのリーダーID

# フォロワーサーボの初期ゴールポジション
initial_goal_positions = [2047] * arm_dim

# Create controller instances
controller_follower = DynamixelController(constants.FOLLOWER1, BAUDRATE)
controller_leader = DynamixelController(constants.LEADER1, BAUDRATE)

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
    initial_goal_positions,
    ADDR_XL_GOAL_POSITION  # ここにアドレスを追加
)
# Set initial positions for follower servos
success = controller_leader.sync_write_goal_position(
    leader1_ids,
    initial_goal_positions,
    ADDR_XL_GOAL_POSITION  # ここにアドレスを追加
)
if success:
    print("サーボへの初期値の反映に成功しました。")
else:
    print("サーボへの初期値の反映に失敗しました。")


