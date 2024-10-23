import argparse

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--method', type=str, default='DC', help='DC/CAFE/IDM/Ours/Test/Continual')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--num_task', type=int, default=5, help='random seed')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--n_ipc', type=int, default=1, help='image(s) per class')
parser.add_argument('--init', type=str, default='random', help='initialization method')
parser.add_argument('--factor', type=int, default=2, help='append image factor')

parser.add_argument('--epochs', type=int, default=300, help='epochs to train a model with synthetic data')
parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') 
# S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')

parser.add_argument('--iteration', type=int, default=2000, help='training iterations')
parser.add_argument('--batchsize', type=int, default=256, help='batch size for real data')

parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--path', type=str, default='save', help='path to save results')
parser.add_argument('--log_level', type=str, choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'), default='INFO', help='path to save results')

parser.add_argument('--verbose', action='store_true', default=False, help='Make the log verbose or not')
parser.add_argument('--use_real_img', action='store_true', default=False, help='use real images or not')
parser.add_argument('--memory_size', type=int, default=0, help='memory size')

parser.add_argument('--o_iter', type=int, default=1, help='iteration of outer loop')
parser.add_argument('--i_iter', type=int, default=1, help='iteration of inner loop')
parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')

parser.add_argument('--dsa', action='store_true', default=False, help='use DSA or not')
parser.add_argument('--dsa_iter', type=int, default=1, help='DSA iterations')

# Distributed
parser.add_argument('--dist_url', default='tcp://')
parser.add_argument('--dist_backend', default='nccl')

# Ours
parser.add_argument('--use_layer_scale', action='store_true', default=False, help='use layer scale or not')
parser.add_argument('--use_freq_schedule', action='store_true', default=False, help='use freq schedule or not')
parser.add_argument('--use_adversarial', action='store_true', default=False, help='use adversarial or not')
parser.add_argument('--use_feature_importance', action='store_true', default=False, help='use feature importance or not')
parser.add_argument('--use_feature_variance', action='store_true', default=False, help='use feature variance or not')
parser.add_argument('--use_image_matching', action='store_true', default=False, help='use image matching or not')

# CAFE
parser.add_argument('--lambda_1', type=float, default=0.04, help='break outlooper')
parser.add_argument('--lambda_2', type=float, default=0.03, help='break innerlooper')
parser.add_argument('--discrimination_loss_weight', type=float, default=0.01, help='discrimination loss weight')

# IDM
parser.add_argument('--net_generate_interval', type=int, default=30, help='outer loop for network update')
parser.add_argument('--net_num', type=int, default=100, help='outer loop for network update')
parser.add_argument('--net_begin', type=int, default=0, help='outer loop for network update')
parser.add_argument('--net_end', type=int, default=100000, help='outer loop for network update')
parser.add_argument('--net_push_num', type=int, default=1, help='outer loop for network update')
parser.add_argument('--ij_selection', type=str, default='random', help='outer loop for network update')
parser.add_argument('--train_net_num', type=int, default=1, help='outer loop for network update')
parser.add_argument('--aug_num', type=int, default=1, help='outer loop for network update')
parser.add_argument('--fetch_net_num', type=int, default=2, help='outer loop for network update')
parser.add_argument('--ce_weight', type=float, default=0.5, help='outer loop for network update')

# IDC
parser.add_argument('--pt_from', type=int, default=-1, help='pretrain from')
parser.add_argument('--mixup', default='vanilla', type=str, choices=('vanilla', 'cut'), help='mixup method')
parser.add_argument('--mixup_net', default='cut', type=str, choices=('vanilla', 'cut'), help='mixup method for network')
parser.add_argument('--fix_iter', type=int, default=-1, help='fix iteration')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--early', type=int, default=0, help='early iteration')
parser.add_argument('--match', type=str, default='grad', choices=['feat', 'grad'], help='feature or gradient matching')
parser.add_argument('--metric', type=str, default='l1', choices=['mse', 'l1', 'l1_mean', 'l2', 'cos'], help='matching objective')
parser.add_argument('--bias', action='store_true', default=False, help='use matching bias or not')
parser.add_argument('--fc', action='store_true', default=False, help='use matching fc or not')
parser.add_argument('--n_data', type=int, default=500, help='number of data for matching')
parser.add_argument('--beta', type=float, default=1.0, help='alpha for mixup')

# DREAM
parser.add_argument('--interval', type=int, default=10, help='K-means interval')

# DualCondensation
parser.add_argument('--model_2', type=str, default='ConvNet', help='model')
parser.add_argument('--lr_net_2', type=float, default=0.01, help='learning rate for updating network parameters')

parser.add_argument('--feature_matching_in_training', action='store_true', default=False, help='feature matching in training')
parser.add_argument('--feature_matching_in_condensation', action='store_true', default=False, help='feature matching in condensation')
parser.add_argument('--gradient_accumulation', action='store_true', default=False, help='gradient accumulation')
parser.add_argument('--gradient_scale_and_clip', action='store_true', default=False, help='gradient scale and clip')

# MTT
parser.add_argument('--max_start_iteration', type=int, default=25, help='max start iteration')
parser.add_argument('--num_experts', type=int, default=100, help='number of experts')
parser.add_argument('--experts_epoch', type=int, default=3, help='epochs to train a model with synthetic data')