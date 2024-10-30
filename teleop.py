"""
Teleoperation Script

This script initializes communication with the leader and follower Dynamixel controllers,
reads data from the leader servos, processes the data, and sends commands to the follower servos.
"""
import sys
import time
from dynamixel_controller import DynamixelController
import constants 
import argparse

# Constants
BAUDRATE = 1000000
# アドレス定義
ADDR_XL_GOAL_POSITION = 116  # 使用しているモデルに応じて変更

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Teleoperation Script for Dynamixel Servos.')
parser.add_argument('-pair', type=int, choices=[1, 2], default=2,
                    help='Number of servo pairs to use (1 or 2). Defaults to 2.')
args = parser.parse_args()
num_pairs = args.pair

# Servo IDs
arm_dim = constants.ARM_DIM

# サーボIDの設定
follower_ids = list(range(1, arm_dim + 1))  # 1からarm_dimまでのフォロワーID
leader_ids = list(range(1, arm_dim + 1))    # 1からarm_dimまでのリーダーID

# フォロワーサーボの初期ゴールポジション
initial_goal_positions = [2047] * arm_dim

# Create controller instances
controller_followers = []
controller_leaders = []

# Set up controllers for each pair
for pair_index in range(num_pairs):
    # Create controller instances
    follower_port = getattr(constants, f'FOLLOWER{pair_index}')
    leader_port = getattr(constants, f'LEADER{pair_index}')

    controller_follower = DynamixelController(follower_port, BAUDRATE)
    controller_leader = DynamixelController(leader_port, BAUDRATE)

    # Attempt to set up the ports
    if not controller_follower.setup_port():
        sys.exit("フォロワーのポート設定に失敗しました。プログラムを終了します。")
    if not controller_leader.setup_port():
        sys.exit("リーダーのポート設定に失敗しました。プログラムを終了します。")
    controller_follower.enable_torque(follower_ids)

    # Set initial positions for follower servos
    success = controller_follower.sync_write_goal_position(
        follower_ids,
        initial_goal_positions,
        ADDR_XL_GOAL_POSITION  # ここにアドレスを追加
    )
    if success:
        print("サーボへの初期値の反映に成功しました。")
    else:
        print("サーボへの初期値の反映に失敗しました。")

    ids = [6]
    PWM_MODE = 16
    controller_leader.set_operation_mode(ids, PWM_MODE)

    # トルクを再度有効化する
    controller_leader.enable_torque(ids)

    # 設定したPWMが反映されるかを確認するためのデバッグ出力
    goal_pwm = 200  # PWM値を調整
    controller_leader.set_pwm(ids, [goal_pwm])

    # Append controllers to the lists
    controller_followers.append(controller_follower)
    controller_leaders.append(controller_leader)
    
try:
    while True:
        for pair_index in range(num_pairs):
            try:
                # リーダーサーボからデータを読み取る
                results_leader = controller_leaders[pair_index].sync_read_data(leader_ids)

                # フォロワーサーボからデータを読み取る
                results_follower = controller_followers[pair_index].sync_read_data(follower_ids)
                
                torque_over = [False] * arm_dim
                for i, dxl_id in enumerate(follower_ids):
                    data = results_follower[dxl_id]
                    load = data['load']
                    if load > 500 or load < -500:
                        torque_over[i] = True
                    velocity = data['velocity']
                    position = data['position']
                    label = "[Follower]"
                    
                # リーダーの位置データを取得
                goal_positions = [results_leader[dxl_id]['position'] for dxl_id in leader_ids]

                # 必要に応じてマッピングや制限を適用
                mapped_positions = []
                for i, position in enumerate(goal_positions):
                    # 値を範囲内に制限
                    new_pos = max(0, min(position, 4095))
                    # Torque overになっている場合はPositionを更新しない
                    if torque_over[i]:
                        new_pos = results_follower[i+1]['position']  
                    mapped_positions.append(new_pos)


                # フォロワーサーボに新しいゴールポジションを送信
                success = controller_followers[pair_index].sync_write_goal_position(
                    follower_ids,
                    mapped_positions,
                    ADDR_XL_GOAL_POSITION  # アドレスを指定
                )

            except KeyError as e:
                print(f"データ取得中にキーエラーが発生しました: {e}")
            except Exception as e:
                print(f"予期しないエラーが発生しました: {e}")
            #finally:
                #time.sleep(0.1)  # 小さな待機を追加して他のタスクの実行を許可

except KeyboardInterrupt:
    for i in range(num_pairs):
        # Disable torque before exiting
        controller_followers[i].disable_torque(follower_ids)
        controller_leaders[i].disable_torque(leader_ids)
        # Close ports
        controller_followers[i].close_port()
        controller_leaders[i].close_port()
        print("プログラムを終了します。")
