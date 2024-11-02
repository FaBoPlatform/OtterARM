import argparse
from constants import DEFAULT_ARGS
from act.play import eval_bc as imitate_eval_bc 

def main():

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Run AI')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., sort)')
    parser.add_argument('--pairs', type=str, required=True)
    args = parser.parse_args()

    task_name = args.task

    from constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    camera_device_ids = task_config['camera_device_ids']
    policy_class = DEFAULT_ARGS['policy_class']
    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': DEFAULT_ARGS['lr'],
                         'num_queries': DEFAULT_ARGS['chunk_size'],
                         'kl_weight': DEFAULT_ARGS['kl_weight'],
                         'hidden_dim': DEFAULT_ARGS['hidden_dim'],
                         'dim_feedforward': DEFAULT_ARGS['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': DEFAULT_ARGS['ckpt_dir'],
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': DEFAULT_ARGS['lr'],
        'policy_class': policy_class,
        'onscreen_render': DEFAULT_ARGS['onscreen_render'],
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': DEFAULT_ARGS['seed'],
        'temporal_agg': DEFAULT_ARGS['temporal_agg'],
        'camera_names': camera_names,
        'camera_device_ids': camera_device_ids,
        'real_robot': True
    }

    print(config)

    ckpt_names = [f'policy_best.ckpt']
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = imitate_eval_bc(config, ckpt_name, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    print()

if __name__ == '__main__':
    main()