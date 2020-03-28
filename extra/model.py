from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

from extra.utils import *
from extra.ADNet import *

class lowlight_enhance(object):
    def __init__(self, sess, is_training):                      # 构造函数初始化，传入会话流图
        self.sess = sess

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')      #占位符定义,喂数据
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        true_noise = self.input_low - self.input_high

        noise, clean = ADNet(self.input_low, is_training=is_training)

        self.output_clean = clean

        self.loss_Denoising = tf.reduce_mean(tf.square(noise - true_noise))   # MSE

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Denoising = [var for var in tf.global_variables() if 'ADNet' in var.name]
        self.var_train_Denoising = [var for var in tf.trainable_variables() if 'ADNet' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_Denoising = optimizer.minimize(self.loss_Denoising, var_list=self.var_train_Denoising)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Denoising = tf.train.Saver(var_list = self.var_Denoising)

        print("[*] Initialize model successfully...")


    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            result_1, result_2 = self.sess.run([self.output_clean, self.input_low], feed_dict={self.input_low: input_low_eval})
            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)


    def data_generator(self, lines, batch_size):
        '''data generator for fit_generator'''
        n = len(lines)  # 训练标签数，也即训练数据集数量
        i = 0  # 当读完一遍数据集后打乱数据集（打乱标签顺序）
        while True:
            low_data = []
            high_data = []
            for b in range(batch_size):  # 循环读取一个batch的数据
                if i == 0:
                    np.random.shuffle(lines)

                name = lines[i].split('\\')[-1]
                b_name = lines[i].split('\\')[-2]
                p_name = b_name.split('/')[2]
                low_path = './data/'+ p_name + '/low/' + name
                high_path = './data/'+ p_name + '/high/' + name

                low_image = Image.open(low_path)
                low_image = np.array(low_image,dtype='float32')/255.0
                high_image = Image.open(high_path)
                high_image = np.array(high_image,dtype='float32')/255.0

                low_data.append(low_image)
                high_data.append(high_image)
                i = (i + 1) % n
            low_data = np.array(low_data)
            high_data = np.array(high_data)
            yield low_data, high_data

    def train(self, train_low_data_names, train_high_data_names, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # load pretrained model
        train_op = self.train_op_Denoising
        train_loss = self.loss_Denoising
        saver = self.saver_Denoising

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()

        generator = self.data_generator(train_low_data_names, batch_size)  # 产生生成器对象
        '''patch的作用'''
        for epoch in range(start_epoch, epoch):                    #遍历总epoch
            for batch_id in range(start_step, numBatch):           #遍历一次epoch中所有的batch
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")   # 传入的不是整张图片，是patch
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

                train_low_data, train_high_data= next(generator)               #迭代器读取

                for patch_id in range(batch_size):                 #遍历一次batch中所有的图片，并进行patch切片训练，同时增强数据
                    h, w, _ = train_low_data[patch_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)               #增强数据
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[patch_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[patch_id][x : x+patch_size, y : y+patch_size, :], rand_mode)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "model-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Denoising, _ = self.load(self.saver_Denoising, './model/Denoising')
        if load_model_status_Denoising:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            print(input_low_test.shape)
            dst = self.sess.run(self.output_clean, feed_dict = {self.input_low: input_low_test})

            save_images(os.path.join(save_dir, name + "_dst." + suffix), dst)

