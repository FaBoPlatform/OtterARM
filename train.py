# train.py

import argparse
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import TRAIN_CONFIG, TASK_CONFIGS, STATE_DIM, PAIR
from act.utils import load_data, compute_dict_mean, set_seed, detach_dict
from act.policy import ACTPolicy, CNNMLPPolicy
from act.train import train_bc, make_policy, make_optimizer  # 必要な関数をインポート

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args_parsed = parser.parse_args()

    set_seed(1)
    # コマンドライン引数
    task_name = args_parsed.task
    num_epochs = args_parsed.epochs

    # タスク設定の取得
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # トレーニング設定の読み込み
    args = TRAIN_CONFIG.copy()
    args['task_name'] = task_name
    args['num_epochs'] = num_epochs

    is_eval = args.get('eval', False)
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args.get('onscreen_render', False)
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    seed = args.get('seed', 0)

    # 固定パラメータ
    state_dim = STATE_DIM  # constants.py から読み込み
    pair = PAIR  # constants.py から読み込み
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'state_dim': state_dim*pair,  # state_dim を追加
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': args['lr'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
            'state_dim': state_dim*pair,  # state_dim を追加
        }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim*pair,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'temporal_agg': args.get('temporal_agg', False),
        'camera_names': camera_names,
        'real_robot': True,  # 実ロボットを想定
    }

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # データセット統計情報を保存
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # 最良のチェックポイントを保存
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

if __name__ == '__main__':
    main()
