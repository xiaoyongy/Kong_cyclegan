import tensorflow as tf
import cv2
import numpy as np
import imageio
from utils import scale_back
from PIL import Image

class Sobel():
    def input_setup(self):
        # 获取符合正则表达式的文件列表
        filenames_A = tf.io.match_filenames_once("D:/note/cyclegan/Kong_cycleganMask/original/*.jpg")  # The name tf.train.match_filenames_once is deprecated

        self.queue_length_A = tf.size(filenames_A)

        filename_queue_A = tf.train.string_input_producer(filenames_A)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)

        self.image_A = tf.subtract(
            tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [250, 200]), 127.5), 1)

    def input_read(self, sess):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num_files_A = sess.run(self.queue_length_A)
        self.A_input = np.zeros((1, 1, 250, 200, 1))
        for i in range(1):
            image_tensor = sess.run(self.image_A[:, :, 0])
            self.A_input[i] = image_tensor.reshape((1, 250, 200, 1))

        coord.request_stop()
        coord.join(threads)

    def struct(self, input_A):
        # image = tf.image.rgb_to_yuv(input_A)
        # input_map_scale = image[:, :, :, 0]

        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

        # Shape = height x width.
        # image = tf.placeholder(tf.float32, shape=[None, None])

        # Shape = 1 x height x width x 1.

        # image_resized = tf.expand_dims(input_A, 3)

        filtered_x = tf.nn.conv2d(input_A, sobel_x_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')
        filtered_y = tf.nn.conv2d(input_A, sobel_y_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')

        G = tf.sqrt(tf.square(filtered_x) + tf.square(filtered_y))
        G_tag = ((G - tf.reduce_min(G)) / (tf.reduce_max(G) - tf.reduce_min(G)))
        return G_tag

    def train(self):
        self.input_setup()
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        input_A = tf.placeholder(tf.float32, [1, 250, 200, 1], name="input_A")
        Sobel = self.struct(input_A)
        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)
            # image_raw_data = tf.gfile.FastGFile('female.jpg', 'rb').read()
            # image_raw_data = cv2.imread('female.jpg')

            # image_yuv = tf.image.rgb_to_yuv(image)
            show_pic = sess.run(Sobel, feed_dict={input_A: self.A_input[0]})
            show_pic_end = scale_back(1-show_pic)


            picture_merge = show_pic_end.reshape((show_pic_end.shape[1], show_pic_end.shape[2]))
            im = Image.fromarray(picture_merge)
            imageio.imwrite('a_picture.jpg', im)


sobel = Sobel()
sobel.train()
