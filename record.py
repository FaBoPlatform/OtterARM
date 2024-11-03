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

def save_data(num_pairs, cap, total_state_dim, dataset_dir, episode_len, episode_name, data_storage_list, camera_device_ids, camera_names, width, height, RECORD):
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
        dset_qpos = obs.create_dataset('qpos', (episode_len, total_state_dim))
        dset_qvel = obs.create_dataset('qvel', (episode_len, total_state_dim))
        dset_action = root.create_dataset('action', (episode_len, total_state_dim))


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
                    velocities = []
                    actions = []

                    for pair_index in range(num_pairs):
                        data_storage = data_storage_list[pair_index]
                        # サーボIDの昇順にデータを取得
                        for dxl_id in sorted(data_storage.keys()):
                            value = data_storage[dxl_id]
                            position_value = normalize_pos(value['position'])
                            velocity_value = normalize_vel(value['velocity'])
                            positions.append(position_value)
                            velocities.append(velocity_value)
                            action_value = normalize_pos(value['position'])
                            actions.append(action_value)

                    # データが不足している場合は0で埋める
                    positions = np.array(positions + [0] * (total_state_dim - len(positions)))
                    velocities = np.array(velocities + [0] * (total_state_dim - len(velocities)))
                    actions = np.array(actions + [0] * (total_state_dim - len(actions)))

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


def run_sync_device(num_pairs, controller_leaders, controller_followers, leader_ids, follower_ids, data_storage_list):
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
            for pair_index in range(num_pairs):
                # リーダーサーボからデータを読み取る
                results_leader = controller_leaders[pair_index].sync_read_data(leader_ids)
                if not results_leader:
                    tqdm.write(" [Leader ARM] リーダーのデータ読み取りに失敗しました。")
                    return

                # フォロワーサーボからデータを読み取る
                results_follower = controller_followers[pair_index].sync_read_data(follower_ids)
                if not results_follower:
                    tqdm.write(" [Follower ARM] フォロワーのデータ読み取りに失敗しました。")
                    return

                torque_over = [False] * state_dim
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

                # データを保存
                data_storage = data_storage_list[pair_index]
                for dxl_id in follower_ids:
                    data = results_follower[dxl_id]
                    data_storage[dxl_id] = {
                        'position': data['position'],
                        'velocity': data['velocity'],
                        'load': data['load']
                    }
                    label = "[Follower]"
                    desc = f"{label:<12} ID: {dxl_id:<2} Load: {load:<10} Velocity: {velocity:<12} Position: {position:<5}"
                    servo_bars[('Follower', dxl_id)].set_description(desc)
                    servo_bars[('Follower', dxl_id)].refresh()
                    # データを保存

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
                        help='使用するサーボペアの数 (1または2)。デフォルトは2。')
    args = parser.parse_args()
    num_pairs = constants.PAIR

    # タスク設定を取得
    task_name = args.task
    if task_name not in constants.TASK_CONFIGS:
        print(f"エラー: タスク '{task_name}' は constants.py に定義されていません。")
        sys.exit(1)
    task_config = constants.TASK_CONFIGS[task_name]
    buadrate = constants.BAUDRATE

    state_dim = constants.STATE_DIM
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config.get('camera_names', ['high'])  # デフォルトで 'high' を使用
    camera_device_ids = task_config.get('camera_device_ids', [0])  # デフォルトで 'high' を使用
    width = task_config.get('width', 640)
    height = task_config.get('height', 480) 
    num_episodes = args.num  # 繰り返す回数

    # サーボIDの設定
    follower_ids = list(range(1, state_dim + 1))  # 1からstate_dimまでのフォロワーID
    leader_ids = list(range(1, state_dim + 1))    # 1からstate_dimまでのリーダーID
    terminate_event = threading.Event()  # スレッドの終了を通知するイベント

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



            # データ保存用のリストを定義
            data_storage_list = [{} for _ in range(num_pairs)]  # 各ペアのデータを保存する辞書のリスト

            # 全体のアーム次元数を計算
            total_state_dim = state_dim * num_pairs

            RECORD = True

            # エピソード番号を取得
            episode_number = get_episode_number(dataset_dir)
            episode_name = f"{episode_number}"

            print(f"エピソード {episode + 1}/{num_episodes} を収集します。エピソード番号: {episode_name}")

            # 終了イベントをクリア
            terminate_event.clear()

            # データ保存用のスレッドを作成
            data_thread = threading.Thread(
                target=save_data,
                args=(num_pairs, cameras, total_state_dim, dataset_dir, episode_len, episode_name, data_storage_list, camera_device_ids, camera_names, width, height, RECORD)
            )

            # デバイス同期用のスレッドを作成
            controll_thread = threading.Thread(
                target=run_sync_device,
                args=(num_pairs, controller_leaders, controller_followers, leader_ids, follower_ids, data_storage_list)
            )

            # スレッドを開始
            data_thread.start()
            controll_thread.start()

            # スレッドの終了を待機
            controll_thread.join()
            data_thread.join()

            for pair_index in range(num_pairs):
                # 終了前にトルクを無効化
                controller_followers[pair_index].disable_torque(follower_ids)
                controller_leaders[pair_index].disable_torque(leader_ids)

            print(f"エピソード {episode + 1}/{num_episodes} の収集が完了しました。")

            # 最後のエピソードでなければカウントダウンを実行
            if episode < num_episodes - 1:
                countdown(10)  # 10秒間のカウントダウン

    except KeyboardInterrupt:
        print("プログラムが中断されました。終了します...")
        terminate_event.set()
        thread.join()
        data_thread.join()
        # トルクを無効化
        controller_follower.disable_torque(follower1_ids)
        controller_leader.disable_torque(leader1_ids)

    finally:
        # カメラデバイスを解放
        release_cameras(*cameras.values())
