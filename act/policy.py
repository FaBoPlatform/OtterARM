import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import argparse

from act.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

# 追加
def get_args_parser():
    parser = argparse.ArgumentParser('DETR training and evaluation script', add_help=False)
    # 学習パラメータ
    parser.add_argument('--lr', default=1e-4, type=float)  # 上書きされる
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 上書きされる
    parser.add_argument('--batch_size', default=2, type=int)  # 使用されない
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)  # 使用されない
    parser.add_argument('--lr_drop', default=200, type=int)  # 使用されない
    parser.add_argument('--clip_max_norm', default=0.1, type=float,  # 使用されない
                        help='gradient clipping max norm')

    # モデルパラメータ
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,  # 上書きされる
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', default=False,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list,  # 上書きされる
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,  # 上書きされる
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,  # 上書きされる
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,  # 上書きされる
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,  # 上書きされる
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,  # 上書きされる
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int,  # 上書きされる
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=False,
                        help="Train segmentation head if the flag is provided")

    # 他の必要な引数を追加
    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', default='ACT', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', default='default_task', type=str, help='task_name')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--num_epochs', default=100, type=int, help='num_epochs')
    parser.add_argument('--kl_weight', default=1, type=int, help='KL Weight')
    parser.add_argument('--chunk_size', default=100, type=int, help='chunk_size')
    parser.add_argument('--temporal_agg', action='store_true')

    # ロボットアームの関節数
    parser.add_argument('--state_dim', default=6, type=int, help='state dimension (number of robot joints)')

    return parser

class ACTPolicy(nn.Module):
    def __init__(self, args_override, device):
        super().__init__()
        self.device = device
        # train.pyからも呼び出せるように、detr/main.py の get_args_parser を使用して args を取得
        args = get_args_parser().parse_args([])
        for k, v in args_override.items():
            setattr(args, k, v)
        model, optimizer = build_ACT_model_and_optimizer(args, self.device)
        self.model = model.to(self.device)
        #self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override, device):
        super().__init__()
        self.device = device
        # train.pyからも呼び出せるように、detr/main.py の get_args_parser を使用して args を取得
        args = get_args_parser().parse_args([])
        for k, v in args_override.items():
            setattr(args, k, v)
        model, optimizer = build_CNNMLP_model_and_optimizer(args, self.device)
        
        #self.model = model # decoder
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
