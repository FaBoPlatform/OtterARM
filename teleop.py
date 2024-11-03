"""
テレオペレーションスクリプト

リーダーとフォロワーのDynamixelコントローラー間の通信を初期化し、
リーダーのサーボからデータを読み取り、処理し、フォロワーのサーボにコマンドを送信します。
"""

import sys
from dynamixel_controller import DynamixelController
import constants
import argparse

def main():
    # 転送速度
    buadrate = constants.BAUDRATE

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Dynamixelサーボ用テレオペレーションスクリプト')
                        help='使用するサーボペアの数 (1または2)。デフォルトは2。')
    args = parser.parse_args()
    num_pairs = constants.PAIR

    # サーボIDの設定
    state_dim = constants.STATE_DIM
    follower_ids = list(range(1, state_dim + 1))  # フォロワーIDリスト
    leader_ids = list(range(1, state_dim + 1))    # リーダーIDリスト

    # コントローラーのインスタンス作成
    controller_followers = []
    controller_leaders = []

    # 各ペアのコントローラーを設定
    for pair_index in range(num_pairs):
        follower_port = getattr(constants, f'FOLLOWER{pair_index}')
        leader_port = getattr(constants, f'LEADER{pair_index}')

        controller_follower = DynamixelController(follower_port, buadrate)
        controller_leader = DynamixelController(leader_port, buadrate)

        # ポートの設定
        if not controller_follower.setup_port():
            sys.exit("フォロワーのポート設定に失敗しました。プログラムを終了します。")
        if not controller_leader.setup_port():
            sys.exit("リーダーのポート設定に失敗しました。プログラムを終了します。")
        controller_follower.enable_torque(follower_ids)

        # リーダーをPWMモードに設定
        ids = [6]
        PWM_MODE = 16
        controller_leader.set_operation_mode(ids, PWM_MODE)

        # トルクを再度有効化する
        controller_leader.enable_torque(ids)

        # 設定したPWMが反映されるかを確認するためのデバッグ出力
        goal_pwm = 200  # PWM値を調整
        controller_leader.set_pwm(ids, [goal_pwm])

        # コントローラーをリストに追加
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

                    torque_over = [False] * state_dim
                    for i, dxl_id in enumerate(follower_ids):
                        data = results_follower[dxl_id]
                        load = data['load']
                        if load > 500 or load < -500:
                            torque_over[i] = True

                    # リーダーの位置データを取得
                    goal_positions = [results_leader[dxl_id]['position'] for dxl_id in leader_ids]

                    # マッピングと制限を適用
                    mapped_positions = []
                    for i, position in enumerate(goal_positions):
                        # 値を範囲内に制限
                        new_pos = max(0, min(position, 4095))
                        # トルクオーバーの場合は位置を更新しない
                        if torque_over[i]:
                            new_pos = results_follower[follower_ids[i]]['position']
                        mapped_positions.append(new_pos)

                    # フォロワーサーボに新しいゴールポジションを送信
                    controller_followers[pair_index].sync_write_goal_position(
                        follower_ids,
                        mapped_positions,
                    )

                except KeyError as e:
                    print(f"データ取得中にキーエラーが発生しました: {e}")
                except Exception as e:
                    print(f"予期しないエラーが発生しました: {e}")

    except KeyboardInterrupt:
        for pair_index in range(num_pairs):
            # 終了前にトルクを無効化
            controller_followers[pair_index].disable_torque(follower_ids)
            controller_leaders[pair_index].disable_torque(leader_ids)
            # ポートを閉じる
            controller_followers[pair_index].close_port()
            controller_leaders[pair_index].close_port()
        print("プログラムを終了します。")

if __name__ == "__main__":
    main()
