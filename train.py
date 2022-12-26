import tensorflow as tf
from model.model import DepthAwareNet, ParameterizedNet, BackboneSharedParameterizedNet, Kitti3DPredictor, Kitti3DBP
from model.loss import L2DepthLoss, L2NormRMSE
from solver.optimizer import OptimizerFactory
import argparse
# import torch

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real', 'kitti', 'kittiBP'], default='kittiBP')
parser.add_argument('-m', '--training-mode', type=str, choices=['normal', 'parameterized', 'shared', 'kitti'], default='shared')
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-j', '--jobs', type=int,  default=8)

args = parser.parse_args()

# input_shape = (180, 320)
input_shape = (180, 595)

################################
#   Define data and dataloader #
################################
if args.data_size == 'big':
    train_path = "./train_new.csv"
    val_path = "./val_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/data"
elif args.data_size == 'kitti':
    train_path = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/train.csv"
    val_path = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/test.csv"
    img_directory = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/images"
    
elif args.data_size == 'kittiBP':
    train_path = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/train.csv"
    val_path = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/test.csv"
    img_directory = "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/images"
else:
    train_path = "./train588_50_new.csv"
    val_path = "./val588_50_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/50imperpose/full/"




if args.training_mode =='normal':
    from data.parallel_dataset import Dataset, DataLoader
else:
    from data.parameterized_parallel_dataset import Dataset, DataLoader
    
# dataset = Dataset(train_path, img_directory, input_shape)
train_dataset = Dataset(train_path, img_directory, input_shape)
val_dataset = Dataset(val_path, img_directory, input_shape)

# split_ratio = 0.8
# train_size = int(split_ratio * len(dataset))
# val_size = len(dataset) - train_size
# train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)
val_loader = DataLoader(val_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)



################
# Define model #
################
if args.training_mode =='parameterized':
    net = ParameterizedNet(num_ext_conv=1)
elif args.training_mode == 'shared':
    net = BackboneSharedParameterizedNet(num_ext_conv=1)
elif args.data_size == 'kitti':
    net = Kitti3DPredictor(num_ext_conv=1)
elif args.data_size == 'kittiBP':
    net = Kitti3DBP(num_ext_conv=1)
else:
    net = DepthAwareNet(num_ext_conv=0)

net.build(input_shape=(None, input_shape[0], input_shape[1], 1))
# inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], 1))
# _ = net.call(inputs)
net.summary()
#######################
# Define loss function#
#######################
USE_MSE = True
if USE_MSE:
    dist_loss_fn = tf.keras.losses.MeanSquaredError()
    depth_loss_fn = tf.keras.losses.MeanSquaredError()
else  :
    dist_loss_fn = L2NormRMSE()
    depth_loss_fn = L2DepthLoss()


#######################
# Define optimizer#
#######################
factory = OptimizerFactory(lr=1e-3, use_scheduler=False)
optimizer = factory.get_optimizer()

#trainer and train
if args.training_mode =='normal':
    from solver.trainer import Trainer
else:
    from solver.parameterized_trainer import Trainer
    
trainer = Trainer(train_loader, val_loader=val_loader,
                    model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                    optimizer=optimizer,
                    log_path='/home/dung/ivsr-s2p/log/training_kitti24012.txt', savepath='/home/dung/ivsr-s2p/ivsr_weights/training_kitti1507',
                    use_mse=USE_MSE)

_  = trainer.train(30, save_checkpoint=True, early_stop=True)
#trainer.save_model()

