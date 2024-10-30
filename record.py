"""
テレオペレーションスクリプト

このスクリプトは、リーダーおよびフォロワーのDynamixelコントローラとの通信を初期化し、
リーダーのサーボからデータを読み取り、データを処理し、
フォロワーのサーボにコマンドを送信します。
また、カメラからの映像を取得し、データを保存します。
"""

import sys
import time
import threading
import os
import cv2
import h5py
import numpy as np
from dynamixel_controller import DynamixelController  # 自作のDynamixelコントローラクラスをインポート
from tqdm import tqdm  # tqdmをインポート
import argparse  # コマンドライン引数を処理するために追加
import constants  # constants.py をインポート
import re  # 正規表現モジュールをインポート


# 定数の設定
BAUDRATE = 1000000  # 通信速度（ボーレート）
# アドレス定義
ADDR_XL_GOAL_POSITION = 116  # ゴールポジションのアドレス（使用しているモデルに応じて変更）
# 動作モードの定数値
CURRENT_CONTROL_MODE = 0
VELOCITY_CONTROL_MODE = 1
POSITION_CONTROL_MODE = 3
EXTENDED_POSITION_CONTROL_MODE = 4
CURRENT_BASED_POSITION_CONTROL_MODE = 5
PWM_CONTROL_MODE = 16  # 既に定義済み

def release_cameras(*caps):
    """
    カメラデバイスを解放する関数。

    :param caps: 解放するカメラデバイスのリスト
    """
    for cap in caps:
        if cap is not None:
            cap.release()
            # print(f"Camera {cap} has been released.")

def setup_camera(device_id, width=640, height=480):
    """
    カメラデバイスを初期化し、指定した解像度に設定する関数。

    :param device_id: カメラデバイスのID
    :param width: フレームの幅
    :param height: フレームの高さ
    :return: 初期化されたカメラデバイス（成功時）、またはNone（失敗時）
    """
    # カメラデバイスを開く
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"デバイスID {device_id} のカメラを開けませんでした。")
        return None

    # フレームの解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # フォーマット設定（必要に応じてコメントアウト）
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    return cap

def normalize_pos(value):
    return (value / 2048 - 1) * 3.14

def normalize_vel(value):
    return value * 3.14 / 2048

def save_data(cap, arm_dim, dataset_dir, episode_len, episode_name, data_storage, camera_device_ids, camera_names, width, height, RECORD):
    """
    データを保存する関数。

    :param dataset_dir: データセットを保存するディレクトリ
    :param episode_len: エピソードの長さ（フレーム数）
    :param data_storage: データを保存する辞書
    :param episode_name: エピソードの名前
    :param camera_device_ids: カメラデバイスのIDリスト
    :param camera_names: カメラの名前のリスト
    :param RECORD: 録画を開始するフラグ
    """
    # データセットのパスを作成
    dataset_path = f"{dataset_dir}/episode_{episode_name}"
    # ディレクトリが存在しない場合は作成
    os.makedirs(dataset_dir, exist_ok=True)
    COMPRESS = False  # 圧縮の設定（必要に応じて変更）

    # HDF5ファイルを作成してデータを保存
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 属性の設定
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
            
        # データセットの作成（固定サイズ、chunksとmaxshapeを指定しない）
        dset_qpos = obs.create_dataset('qpos', (episode_len, arm_dim))
        dset_qvel = obs.create_dataset('qvel', (episode_len, arm_dim))
        dset_action = root.create_dataset('action', (episode_len, arm_dim))


        # 画像データセットの作成
        dset_image = {}
        for cam_name in camera_names:
            dset_image[f'{cam_name}'] = image.create_dataset(
                f'{cam_name}', (0, height, width, 3), dtype='uint8',
                chunks=(1, height, width, 3), maxshape=(episode_len, height, width, 3)
            )

        i = 0  # フレームカウンターの初期化
        with tqdm(total=episode_len, desc="進行状況", unit="フレーム") as pbar:
            try:
                while not terminate_event.is_set():
                    if not RECORD:
                        time.sleep(1)
                        continue

                    ret = {}    # フレーム取得の成否を格納
                    frame = {}  # フレームデータを格納
                    image_frames = {}  # RGB画像を保存
                    # 各カメラからフレームを取得
                    for a in camera_device_ids:
                        ret[a], frame[a] = cap[a].read()
                        if not ret[a]:
                            print(f"カメラ {a} (デバイスID: {camera_device_ids[a]}) からフレームの取得に失敗しました。")
                            # 取得失敗時は黒い画像を用意
                            frame[a] = np.zeros((height, width, 3), dtype=np.uint8)
                        image_frames[a] = cv2.cvtColor(frame[a], cv2.COLOR_BGR2RGB)

                    # サーボのデータを取得
                    positions = []
                    velocities = []  # 速度データ用のリスト
                    actions = []
                    for key in sorted(data_storage.keys()):
                        value = data_storage[key]
                        position_value = normalize_pos(value['position'])
                        velocity_value = normalize_vel(value['velocity'])
                        positions.append(position_value)
                        velocities.append(velocity_value)
                        action_value = normalize_pos(value['position'])
                        actions.append(action_value)

                    # データが不足している場合は0で埋める
                    positions = np.array(positions + [0] * (arm_dim - len(positions)))
                    velocities = np.array(velocities + [0] * (arm_dim - len(velocities)))
                    actions = np.array(actions + [0] * (arm_dim - len(actions)))

                    # データセットのサイズを拡張
                    new_size = i + 1
                    
                    for a, cam_name in enumerate(camera_names):
                        dset_image[f'{cam_name}'].resize(new_size, axis=0)

                    # データを保存
                    for a, cam_name in enumerate(camera_names):
                        dset_image[f'{cam_name}'][i] = image_frames[camera_device_ids[a]]  # 画像フレームを保存

                    dset_qpos[i] = positions  # ポジションデータを保存
                    dset_qvel[i] = velocities  # 速度データを保存
                    dset_action[i] = actions  # アクションデータを保存

                    i += 1  # フレームカウンターをインクリメント

                    # プログレスバーを更新
                    pbar.update(1)

                    # エピソードの長さに達したら終了
                    if pbar.n >= episode_len:
                        print(f"データ収集が完了しました。フレーム数: {pbar.n}")
                        break

                    time.sleep(0.03)  # 少し待機

                terminate_event.set()  # 終了イベントをセット
            except KeyboardInterrupt:
                print("プロセスが中断されました。クリーンアップします...")
                terminate_event.set()
            finally:
                # カメラデバイスを解放（ループの外で行うためコメントアウト）
                # release_cameras(*cap.values())
                pass


def run_sync_device(controller_leader1, controller_follower1, leader_ids, follower_ids):
    global data_storage

    # 各サーボのtqdmインスタンスを保持する辞書
    servo_bars = {}

    # メインのプログレスバーの行数を取得
    main_bar_lines = 1  # プログレスバーは1行を使用

    # Leaderサーボのプログレスバーを作成
    for idx, dxl_id in enumerate(leader_ids):
        position = main_bar_lines + idx  # Leaderサーボはプログレスバーの直下に表示
        label = 'Leader'
        servo_bars[(label, dxl_id)] = tqdm(
            total=1,
            bar_format='{desc}',
            position=position,
            leave=True
        )

    # 区切り線を挿入
    separator_position = main_bar_lines + len(leader_ids)
    separator_bar = tqdm(
        total=1,
        bar_format='{desc}',
        position=separator_position,
        leave=True
    )
    separator_bar.set_description('-' * 50)
    separator_bar.refresh()

    # Followerサーボのプログレスバーを作成
    for idx, dxl_id in enumerate(follower_ids):
        position = main_bar_lines + len(leader_ids) + 1 + idx  # +1は区切り線の分
        label = 'Follower'
        servo_bars[(label, dxl_id)] = tqdm(
            total=1,
            bar_format='{desc}',
            position=position,
            leave=True
        )

    while not terminate_event.is_set():
        try:
            # リーダーサーボからデータを読み取る
            results_leader = controller_leader1.sync_read_data(leader_ids)
            if not results_leader:
                tqdm.write(" [Leader ARM] リーダーのデータ読み取りに失敗しました。")
                return

            # フォロワーサーボからデータを読み取る
            results_follower = controller_follower1.sync_read_data(follower_ids)
            if not results_follower:
                tqdm.write(" [Follower ARM] フォロワーのデータ読み取りに失敗しました。")
                return

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
            success = controller_follower1.sync_write_goal_position(
                follower_ids,
                mapped_positions,
                ADDR_XL_GOAL_POSITION  # アドレスを指定
            )

            # データを保存し、表示を更新
            for dxl_id in leader_ids:
                data = results_leader[dxl_id]
                load = data['load']
                velocity = data['velocity']
                position = data['position']
                label = "[Leader]"
                desc = f"{label:<12} ID: {dxl_id:<2} Load: {load:<10} Velocity: {velocity:<12} Position: {position:<5}"
                servo_bars[('Leader', dxl_id)].set_description(desc)
                servo_bars[('Leader', dxl_id)].refresh()

            # 区切り線は固定なので更新不要

            for dxl_id in follower_ids:
                data = results_follower[dxl_id]
                load = data['load']
                velocity = data['velocity']
                position = data['position']
                label = "[Follower]"
                desc = f"{label:<12} ID: {dxl_id:<2} Load: {load:<10} Velocity: {velocity:<12} Position: {position:<5}"
                servo_bars[('Follower', dxl_id)].set_description(desc)
                servo_bars[('Follower', dxl_id)].refresh()
                # データを保存
                data_storage[dxl_id] = {'position': position, 'velocity': velocity, 'load': load}

            if not success:
                tqdm.write("サーボへの値の反映に失敗しました。")

        except Exception as e:
            tqdm.write(f"予期しないエラーが発生しました: {e}")
        finally:
            time.sleep(0.01)

    # ループ終了後にプログレスバーを閉じる
    for bar in servo_bars.values():
        bar.close()
    separator_bar.close()


def get_episode_number(dataset_dir):
    """
    ディレクトリ内のエピソードファイルの数をカウントし、次のエピソード番号を取得する関数。

    :param dataset_dir: データセットのディレクトリ
    :return: エピソード番号（整数）
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        return 0

    files = os.listdir(dataset_dir)
    # 'episode_数字.hdf5' にマッチするファイルを探す
    pattern = re.compile(r'episode_(\d+)\.hdf5')
    episode_numbers = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            episode_numbers.append(int(match.group(1)))

    if episode_numbers:
        return max(episode_numbers) + 1  # 最大の番号に1を足す
    else:
        return 0  # エピソードファイルがない場合は0を返す


def find_available_cameras(max_devices=4):
    """
    利用可能なカメラデバイスIDを探索する関数。

    :param max_devices: 探索する最大デバイスID数
    :return: 利用可能なカメラデバイスIDのリスト
    """
    available_cameras = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            print(f"カメラがデバイスID {device_id} で認識されました。")
            available_cameras.append(device_id)
            cap.release()
        else:
            print(f"デバイスID {device_id} のカメラは利用できません。")
    return available_cameras


def countdown(seconds):
    """
    カウントダウンを表示する関数。

    :param seconds: カウントダウンする秒数
    """
    for i in range(seconds, 0, -1):
        print(f"{i}秒後に次のエピソードが開始されます...")
        time.sleep(1)
    print("START!")


# メイン処理
if __name__ == "__main__":
    # コマンドライン引数を処理
    parser = argparse.ArgumentParser(description='Record Episode Script')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., sort)')
    parser.add_argument('--num', type=int, default=1, help='Number of episodes to record')
    args = parser.parse_args()

    # タスク設定を取得
    task_name = args.task
    if task_name not in constants.TASK_CONFIGS:
        print(f"エラー: タスク '{task_name}' は constants.py に定義されていません。")
        sys.exit(1)
    task_config = constants.TASK_CONFIGS[task_name]

    arm_dim = task_config['arm_dim']
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config.get('camera_names', ['high'])  # デフォルトで 'high' を使用
    camera_device_ids = task_config.get('camera_device_ids', [0])  # デフォルトで 'high' を使用
    width = task_config.get('width', 640)
    height = task_config.get('height', 480) 
    num_episodes = args.num  # 繰り返す回数

    DEVICENAME_FOLLOWER1 =  constants.FOLLOWER1 # フォロワー1のデバイス名（Windowsでは大文字に注意）
    DEVICENAME_LEADER1 = constants.LEADER1    # リーダー1のデバイス名


    # サーボIDの設定
    follower1_ids = list(range(1, arm_dim + 1))  # 1からarm_dimまでのフォロワーID
    leader1_ids = list(range(1, arm_dim + 1))    # 1からarm_dimまでのリーダーID

    # フォロワーサーボの初期ゴールポジション
    initial_goal_positions = [2047] * arm_dim

    # コントローラインスタンスの作成
    controller_follower = DynamixelController(DEVICENAME_FOLLOWER1, BAUDRATE)
    controller_leader = DynamixelController(DEVICENAME_LEADER1, BAUDRATE)
    terminate_event = threading.Event()  # スレッドの終了を通知するイベント

    # ポートの設定を試みる
    if not controller_follower.setup_port():
        sys.exit("フォロワーのポート設定に失敗しました。プログラムを終了します。")
    if not controller_leader.setup_port():
        sys.exit("リーダーのポート設定に失敗しました。プログラムを終了します。")

    # 利用可能なカメラデバイスIDを確認
    print("カメラをチェックします...")
    available_cameras = find_available_cameras()
    print("利用可能なカメラデバイスID:", available_cameras)

    # カメラデバイスIDを指定（必要に応じて変更）
   

    print(f"カメラデバイスの初期化: {camera_device_ids}, {camera_names}")
    cameras = {}
    for device_id in camera_device_ids:
        cameras[device_id] = setup_camera(device_id, width, height)
        if cameras[device_id] is None:
            print(f"カメラ {device_id} (デバイスID: {device_id}) の初期化に失敗しました。")
        else:
            print(f"カメラ {device_id} (デバイスID: {device_id}) の初期化に成功しました。")

    try:
        for episode in range(num_episodes):
            # データ保存用の辞書
            data_storage = {}
            RECORD = True

            # エピソード番号を取得
            episode_number = get_episode_number(dataset_dir)
            episode_name = f"{episode_number}"

            print(f"エピソード {episode + 1}/{num_episodes} を収集します。エピソード番号: {episode_name}")

            controller_follower.set_operation_mode(follower1_ids, POSITION_CONTROL_MODE)

            # フォロワーのサーボのトルクを有効化
            controller_follower.enable_torque(follower1_ids)

            # フォロワーサーボの初期位置を設定
            success = controller_follower.sync_write_goal_position(
                follower1_ids,
                initial_goal_positions,
                ADDR_XL_GOAL_POSITION  # アドレスを指定
            )
            if success:
                print("サーボへの初期値の反映に成功しました。")
            else:
                print("サーボへの初期値の反映に失敗しました。")

            controller_leader.set_operation_mode(leader1_ids, POSITION_CONTROL_MODE)

            # リーダーのサーボの動作モードを設定
            ids = [arm_dim]  # 操作するサーボのIDリスト
            PWM_MODE = 16  # PWMモードの定数
            controller_leader.set_operation_mode(ids, PWM_MODE)

            # トルクを再度有効化
            controller_leader.enable_torque(ids)

            # デバッグ用にPWM値を設定してみる
            goal_pwm = 200  # PWMの目標値
            controller_leader.set_pwm(ids, [goal_pwm])

            # 終了イベントをクリア
            terminate_event.clear()

            # データ保存用のスレッドを作成
            data_thread = threading.Thread(
                target=save_data,
                args=(cameras, arm_dim, dataset_dir, episode_len, episode_name, data_storage, camera_device_ids, camera_names, width, height, RECORD)
            )

            # デバイス同期用のスレッドを作成
            thread1 = threading.Thread(
                target=run_sync_device,
                args=(controller_leader, controller_follower, leader1_ids, follower1_ids)
            )

            # スレッドを開始
            data_thread.start()
            thread1.start()

            # スレッドの終了を待機
            thread1.join()
            data_thread.join()

            # トルクを無効化
            controller_follower.disable_torque(follower1_ids)
            controller_leader.disable_torque(leader1_ids)

            print(f"エピソード {episode + 1}/{num_episodes} の収集が完了しました。")

            # 最後のエピソードでなければカウントダウンを実行
            if episode < num_episodes - 1:
                countdown(10)  # 10秒間のカウントダウン

    except KeyboardInterrupt:
        print("プログラムが中断されました。終了します...")
        terminate_event.set()
        thread1.join()
        data_thread.join()
        # トルクを無効化
        controller_follower.disable_torque(follower1_ids)
        controller_leader.disable_torque(leader1_ids)

    finally:
        # カメラデバイスを解放
        release_cameras(*cameras.values())
