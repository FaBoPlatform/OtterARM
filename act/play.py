import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from act.utils import load_data # data functions
from act.utils import sample_box_pose, sample_insertion_pose # robot functions
from act.utils import compute_dict_mean, set_seed, detach_dict # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy

from dynamixel_controller import DynamixelController

import collections
import constants
import cv2

import IPython
e = IPython.embed

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

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

def denormalize_pos(radians):
    # Convert radians to position counts (0 to 4095)
    # Assuming the range of motion is from -3.14 to +3.14 radians
    return int((radians / 3.14 + 1) * 2048)

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)

    # 転送速度
    buadrate = constants.BAUDRATE
    # サーボIDの設定
    arm_dim = constants.ARM_DIM
    follower_ids = list(range(1, arm_dim + 1))  # フォロワーIDリスト
    leader_ids = list(range(1, arm_dim + 1))    # リーダーIDリスト

    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    camera_device_ids = config['camera_device_ids']
    onscreen_cam = 'angle'

    # コントローラーのインスタンス作成
    controller_followers = []
    controller_leaders = []
    width = 320
    height = 240
    cameras = {}
    for device_id in camera_device_ids:
        cameras[device_id] = setup_camera(device_id, width, height)
        if cameras[device_id] is None:
            print(f"カメラ {device_id} (デバイスID: {device_id}) の初期化に失敗しました。")
        else:
            print(f"カメラ {device_id} (デバイスID: {device_id}) の初期化に成功しました。")


    # 各ペアのコントローラーを設定
    num_pairs = 1
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

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    env_max_reward = 0
  
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    state_dim = 6
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    
        with torch.inference_mode():
            for t in range(max_timesteps):
                
                ### process previous timestep to get qpos and image_list
                # フォロワーサーボからデータを読み取る
                results_follower = controller_followers[0].sync_read_data(follower_ids)
                # リーダーの位置データを取得
                qpos = [normalize_pos(results_follower[dxl_id]['position']) for dxl_id in follower_ids]
                #print(f"qpos{qpos}")
                qpos = pre_process(qpos)
                #print(f"qpos{qpos}")
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                ret = {}    # フレーム取得の成否を格納
                frame = {}  # フレームデータを格納
                curr_images = []
                for a in camera_device_ids:
                    ret[a], frame[a] = cameras[a].read()
                    curr_image = frame[a]
                    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
                    curr_image = rearrange(curr_image, 'h w c -> c h w')
                    curr_images.append(curr_image)
                curr_image = np.stack(curr_images, axis=0)
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                # Convert target_qpos from radians to position counts
                target_position_counts = [denormalize_pos(angle) for angle in target_qpos]
                #print("followers action!!")
                controller_followers[0].sync_write_goal_position(
                    follower_ids,
                    target_position_counts,
                )

                pass
        
        # 終了前にトルクを無効化
        controller_followers[pair_index].disable_torque(follower_ids)

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return