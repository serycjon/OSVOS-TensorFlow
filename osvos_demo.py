"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataloader import OsvosDataloader
from dataset import Dataset
os.chdir(root_folder)
import datetime
import argparse

parser = argparse.ArgumentParser(description='OSVOS TF implementation.')
parser.add_argument('--seq', type=str, help='DAVIS sequence', default='car-shadow')
parser.add_argument('--train', help='do the finetuning first', action='store_true')

args = parser.parse_args()

# User defined parameters
seq_name = args.seq  # "car-shadow"
gpu_id = 0
train_model = args.train  # False
result_path = os.path.join('Results', seq_name)

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
stamp = datetime.datetime.now().isoformat()
logs_path = os.path.join('models', seq_name, stamp)
model_path = os.path.join('models', seq_name)
max_training_iters = 500

from osvos import osvos_parameters
params = osvos_parameters(batch_size = 1,
                          num_threads = 1,
                          height = 400,
                          width = 600)

with tf.Graph().as_default():

    # loader = OsvosDataloader('DAVIS',
    #                         filenames_file = 'list.txt',
    #                         params = params,
    #                         mode = 'train')

    train_imgs = [os.path.join('/JPEGImages', '480p', seq_name, '00000.jpg') + ' ' + \
                  os.path.join('/Annotations', '480p', seq_name, '00000.png')]
    
    loader = OsvosDataloader('DAVIS',
                            filenames_file = train_imgs,
                            params = params,
                            mode = 'train')

    # Train the network
    if train_model:
        # More training parameters
        learning_rate = 1e-8
        save_step = max_training_iters
        side_supervision = 3
        display_step = 10
        # with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(loader, parent_path, side_supervision, learning_rate, model_path, logs_path, max_training_iters,
                                save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

with tf.Graph().as_default():
    test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
    test_imgs = [os.path.join('/JPEGImages', '480p', seq_name, frame) for frame in test_frames]
    loader = OsvosDataloader('DAVIS',
                             filenames_file = test_imgs,
                             params = params,
                             mode = 'test')

    # Test the network
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join(model_path, seq_name+'.ckpt-'+str(max_training_iters))
        # checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(loader, checkpoint_path, result_path)

# # Show results
# overlay_color = [255, 0, 0]
# transparency = 0.6
# plt.ion()
# for img_p in test_frames:
#     frame_num = img_p.split('.')[0]
#     img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
#     mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
#     mask = mask/np.max(mask)
#     im_over = np.ndarray(img.shape)
#     im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
#     im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
#     im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
#     plt.imshow(im_over.astype(np.uint8))
#     plt.axis('off')
#     plt.show()
#     plt.pause(0.01)
#     plt.clf()
