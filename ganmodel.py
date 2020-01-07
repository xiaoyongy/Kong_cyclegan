# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import scipy.misc as misc
from networklayers import build_generator_resnet_6blocks_d, build_gen_discriminator
from nextbatch import next_batch_dataset, ImagePool
from utils import scale_back, normalize_image, merge
import time
import datetime

"""
修改：
img_width=120, img_height=120
"""
class CycleGan(object):
    
    def __init__(self, epoch=5000, to_restore=False, mse_penalty=100.0, #max_images=50,
                 checkpiont_dir='checkpoint/',
                 lr=0.0001, batch_size=1, img_width=128, img_height=128, img_layer=3, pool_maxsize=50,
                 sample_steps=100,
                 max_images=1000):
        self.mse_penalty = mse_penalty
        self.to_restore = to_restore
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.img_layer = img_layer
        self.sample_steps = sample_steps
        self.checkpiont_dir = checkpiont_dir
        self.pool_maxsize = pool_maxsize
        self.pool = ImagePool(self.pool_maxsize)
        #self.totalnum = max_images
        self.img_size = img_height*img_width
        self.max_images = max_images

    def input_setup(self):

        """
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        :return:
        """
        # 获取符合正则表达式的文件列表
        filenames_A = tf.io.match_filenames_once(
            "F:/dataset/sketches/original/*.jpg")  # The name tf.train.match_filenames_once is deprecated
        self.queue_length_A = tf.size(filenames_A)
        filenames_B = tf.io.match_filenames_once("F:/dataset/sketches/original_sketch/*.jpg")
        self.queue_length_B = tf.size(filenames_B)

        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(
            tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [self.img_height, self.img_width]), 127.5), 1)
        self.image_B = tf.subtract(
            tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [self.img_height, self.img_width]), 127.5), 1)


    def input_read(self, sess):

        """
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        :param sess:
        :return:
        """

        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((self.pool_maxsize, 1, self.img_height, self.img_width, self.img_layer))
        self.fake_images_B = np.zeros((self.pool_maxsize, 1, self.img_height, self.img_width, self.img_layer))

        self.A_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))
        self.B_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))

        for i in range(self.max_images):
            image_tensor = sess.run(self.image_A)
            # print(image_tensor.shape)  # (256, 256, 3)
            if image_tensor.size == self.img_size*self.batch_size*self.img_layer:
                self.A_input[i] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.img_layer))
                # print(self.A_input[i].shape)  # (1, 256, 256, 3)
                # print(self.A_input[i])  # 里面都是1

        for i in range(self.max_images):
            image_tensor = sess.run(self.image_B)
            if image_tensor.size == self.img_size*self.batch_size*self.img_layer:
                self.B_input[i] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.img_layer))

        coord.request_stop()
        coord.join(threads)


    def checkpoint(self, saver, step, sess):
        model_name = "Mustachenet.model"
        model_dir = self.checkpiont_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(sess, os.path.join(model_dir, model_name), global_step=step)
    
    def sample_model(self, rA, rB, epoch, count, sess):
        fake_B, b_mask, fake_A, a_mask = sess.run([self.fake_B, self.B_mask, self.fake_A, self.A_mask],
                                                  feed_dict={self.input_A: rA, self.input_B: rB})
        # out_gen = out_gen*mask + inputgen*(1-mask)
        a1 = scale_back(rA)
        a2 = scale_back(fake_B)
        #a3 = b_mask*255.0
        # a4 = (a2-a1*(1-b_mask))/b_mask  # scale_back(fake_B*b_mask+rA*(1-b_mask))

        b1 = scale_back(rB)
        b2 = scale_back(fake_A)
        #b3 = a_mask*255.0
        # b4 = (b2-b1*(1-a_mask))/a_mask  # scale_back(fake_A*a_mask+rB*(1-a_mask))

        merged_pair = np.concatenate([a1, a2, b1, b2], axis=2)
        merged_pair = merged_pair.reshape((merged_pair.shape[1], merged_pair.shape[2], merged_pair.shape[3]))
        #        print(merged_pair.shape)
        s_dir = 'D:/note/cyclegan/Kong_cycleganMask/samples/'
        
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)
        
        sample_img_path = os.path.join(s_dir, "sample_%02d_%04d.png" % (epoch, count))
        misc.imsave(sample_img_path, merged_pair)



    def buildmodel(self):
        
        self.input_A = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_layer],
                                      name="input_A")
        self.input_B = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_layer],
                                      name="input_B")

        # 输入 real A
        self.fake_B, self.B_mask = build_generator_resnet_6blocks_d(self.input_A, reuse=False, name="g_A2B")
        self.cyc_A, _ = build_generator_resnet_6blocks_d(self.fake_B, reuse=False, AB=False, name="g_B2A")
        # 输入 real B
        self.fake_A, self.A_mask = build_generator_resnet_6blocks_d(self.input_B, reuse=True, AB=False, name="g_B2A")
        self.cyc_B, _ = build_generator_resnet_6blocks_d(self.fake_A, reuse=True, name="g_A2B")
        
        self.DB_fake = build_gen_discriminator(self.fake_B, reuse=False, name="d_B")
        self.DA_fake = build_gen_discriminator(self.fake_A, reuse=False, name="d_A")
        
#        self.fake_rec_A=build_gen_discriminator(self.fake_A,reuse=True,name="d_A")
#        self.fake_rec_B=build_gen_discriminator(self.fake_B,reuse=True,name="d_B")
#        
        # L1
#        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        self.cyc_loss = self.mse_penalty * tf.reduce_mean(tf.squared_difference(self.input_B, self.cyc_B)) + \
                        self.mse_penalty * tf.reduce_mean(tf.squared_difference(self.input_A, self.cyc_A))
        
#        #mse
        # self.cyc_loss=self.mse_penalty *tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.input_B, self.cyc_B), [1, 2, 3]))\
        # +self.mse_penalty *tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.input_A, self.cyc_A), [1, 2, 3]))
#      
        self.g_loss_a2b = tf.reduce_mean(tf.squared_difference(self.DB_fake,tf.ones_like(self.DB_fake)))+self.cyc_loss
        self.g_loss_b2a = tf.reduce_mean(tf.squared_difference(self.DA_fake,tf.ones_like(self.DA_fake)))+self.cyc_loss

        self.g_loss=tf.reduce_mean(tf.squared_difference(self.DB_fake,tf.ones_like(self.DB_fake))) \
                            +tf.reduce_mean(tf.squared_difference(self.DA_fake,tf.ones_like(self.DA_fake))) \
                            +self.cyc_loss
        
                
        self.fake_A_sample = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_layer],  name="fake_A_sample")
        self.fake_B_sample = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_layer],  name="fake_B_sample")
   
        self.DA_real=build_gen_discriminator(self.input_A,reuse=True,name="d_A")
        self.DB_real=build_gen_discriminator(self.input_B,reuse=True,name="d_B")
        
        self.DA_fake_sample = build_gen_discriminator(self.fake_A_sample,reuse=True,name="d_A")
        self.DB_fake_sample = build_gen_discriminator(self.fake_B_sample,reuse=True,name="d_B")

        self.db_loss_real=tf.reduce_mean(tf.squared_difference(self.DB_real,tf.ones_like(self.DB_real)))
        self.db_loss_fake=tf.reduce_mean(tf.squared_difference(self.DB_fake_sample,tf.zeros_like(self.DB_fake_sample)))
        self.db_loss=0.5*(self.db_loss_real+self.db_loss_fake)
        
        self.da_loss_real=tf.reduce_mean(tf.squared_difference(self.DA_real,tf.ones_like(self.DA_real)))
        self.da_loss_fake=tf.reduce_mean(tf.squared_difference(self.DA_fake_sample,tf.zeros_like(self.DA_fake_sample)))
        self.da_loss=0.5*(self.da_loss_real+self.da_loss_fake)
        
        self.d_loss=self.db_loss+self.da_loss
        
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

    def train(self):

        self.input_setup()

        # self.init_data()
        # imga = self.read_and_decode(self.A_dir)
        # imgA_batch, _ = tf.train.shuffle_batch([imga, 1], batch_size=1, capacity=20, min_after_dequeue=10)
        # imgb = self.read_and_decode(self.B_dir)
        # imgB_batch, _ = tf.train.shuffle_batch([imgb, 2], batch_size=1, capacity=20, min_after_dequeue=10)
        self.buildmodel()
        self.model_vars = tf.trainable_variables()
#        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.8)
        g_vars = [var for var in self.model_vars if 'g_' in var.name]
        d_vars = [var for var in self.model_vars if 'd_' in var.name]
       #
       # self.d_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.8).minimize(self.d_loss, var_list=d_vars)
       # self.g_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.8).minimize(self.g_loss, var_list=g_vars)
       # self.clip_DB = [p.assign(tf.clip_by_value(p, -0.04, 0.04)) for p in d_vars]

        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=g_vars)
        
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_time = nowTime

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
#        init=tf.global_variables_initializer()
        saver = tf.train.Saver()

        
        with tf.Session() as sess:
            # 先执行初始化工作
            # sess.run(tf.local_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            sess.run(init)

            self.input_read(sess)

            writer = tf.summary.FileWriter("./logs", sess.graph)
#            Restore the model to run the model from last checkpoint
            if self.to_restore:  # False
                chkpt_fname = tf.train.latest_checkpoint(self.checkpiont_dir)
                saver.restore(sess, chkpt_fname)
        
            if not os.path.exists(self.checkpiont_dir):
                os.makedirs(self.checkpiont_dir)
          
            counter = 1

            g_total_loss = []
            d_total_loss = []
            
#            g_avg=[]
#            d_avg=[]
#            
#            checkgtmp=0.0
            from tkinter import _flatten
            for epoch in range(self.epoch):
                # g_avg.clear()
                for step in range(0, self.max_images):
                    # rA,rB=self.traindata.next_batch(self.batch_size)
                    #rA, rB = sess.run([self.image_A, self.image_B])  # rA,rB并不是list类型
                    #rA=normalize_image(np.array(rA))
#                    print(rA.shape)
                    #rB=normalize_image(np.array(rB))
#                    print(epoch,step)
#                    print(rB.shape)


                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_g, g_T_loss = sess.run([self.fake_A, self.fake_B, self.g_optim,
                                                                       self.g_sum, self.g_loss],
                                                                      feed_dict={self.input_A: self.A_input[step],
                                                                                 self.input_B: self.B_input[step]})
                    writer.add_summary(summary_g, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])
                    
                    # Update D network
                    _, summary_d, d_T_loss = sess.run([self.d_optim, self.d_sum, self.d_loss],  # ,self.clip_DB ,_
                                                      feed_dict={self.input_A: self.A_input[step],
                                                                 self.input_B: self.B_input[step],
                                                                 self.fake_A_sample: fake_A,
                                                                 self.fake_B_sample: fake_B})
                    writer.add_summary(summary_d, counter)
                    counter += 1
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(("Epoch: [%2d] [%4d/%4d], time: %10s, g_T_loss: %.5f, g_D_loss: %.5f" % (
                        epoch, step, self.max_images, nowTime, g_T_loss, d_T_loss)))
                    
                    if counter % self.sample_steps == 0:
                        self.sample_model(self.A_input[step], self.B_input[step], epoch, counter, sess)

                    g_total_loss.append(g_T_loss)
                    d_total_loss.append(d_T_loss)

                # if epoch == 13:
                #     self.sample_steps = 30
                # if epoch == 15:
                #     self.sample_steps = 3
                #
                # if epoch == 16:
                #     break
                
#                if epoch==780:
#                    checkgtmp=gtmp
                
#                if epoch>=800 and checkgtmp>gtmp:
#                   checkgtmp=gtmp
#                   print("Checkpoint: save checkpoint step %d" % epoch)
#                   self.checkpoint(saver,epoch,sess)
            
           
            
            
            #save npy
            
            with open('D:/note/cyclegan/Kong_cycleganMask/npyfile/g_total_loss.npy','wb') as f1:
                np.save(f1, g_total_loss)
            with open('D:/note/cyclegan/Kong_cycleganMask/npyfile/d_total_loss.npy','wb') as f2:
                np.save(f2, d_total_loss)
            
            print("Checkpoint: save checkpoint epoch %d" % epoch)
            self.checkpoint(saver,epoch,sess)

            end_Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('start_time: {0}, and end_time: {1}'.format(start_time, end_Time))
