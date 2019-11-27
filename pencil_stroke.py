import numpy as np
import cv2
import tensorflow as tf
from skimage import transform, color
from scipy import signal
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

def struct(input_A, height, width):
    input_A = tf.squeeze(input_A)
    input_A = input_A[:, :, 0]
    input_A = tf.reshape(input_A, [1, height, width, 1])


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

def gen_stroke(input_map, kernel_size, height, width, stroke_width=0, num_of_direction=8):
    input_gray = color.rgb2yuv(input_map)
    input_map_scale = input_gray[:, :, 0]
    sobelx = cv2.Sobel(input_map_scale, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(input_map_scale, cv2.CV_64F, 0, 1, ksize=5)
    G = np.sqrt(np.square(sobelx) + np.square(sobely))


    basic_ker = np.zeros((kernel_size*2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1, :] = 1
    res_map = np.zeros((height, width, num_of_direction))

    for d in range(num_of_direction):
        ker = transform.rotate(basic_ker, (d * 180)/num_of_direction)


        res_map[:,:, d] = signal.convolve2d(G, ker, mode='same')
    #res_map = tf.Variable(res_map)
    max_pixel_indices_map = np.argmax(res_map, axis=2)
    C = np.zeros_like(res_map)
    for d in range(num_of_direction):
        C[:, :, d] = G * (max_pixel_indices_map == d)
    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_direction):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_direction)
        S_tag_sep[:, :, d] = signal.convolve2d(C[:, :, d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)
    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))

    G_tag_normalized = (G - np.min(G.ravel())) / (np.max(G.ravel()) - np.min(G.ravel()))
    G_tag = 1 - G_tag_normalized
    return G_tag

from scipy.ndimage import filters
# map = cv2.imread('female.jpg')
# map_yuv = color.rgb2yuv(map)
# map_y_ch = map_yuv[:, :, 0]
# gen_stroke_map = gen_stroke(map_y_ch, kernel_size=3, height=250, width=200)
# # # plt.imshow(gen_stroke_map, cmap='gray')
# # # # plt.show()
# # # # # print(gen_stroke)
# # # plt.imsave('female_ske.jpg', gen_stroke_map)
# # #
# # # plt.show()
# cv2.imwrite('female_ske.jpg', gen_stroke_map*255)
# cv2.imshow('sh', gen_stroke_map)
# cv2.waitKey(0)

def fun(img, sess):
    image = tf.image.rgb_to_yuv(tf.cast(img, tf.float32))
    print('image', sess.run(image))
    return sess.run(image[:, :, 0])

# with tf.Session() as sess:
#     image_raw_data_jpg = tf.gfile.FastGFile('female.jpg', 'rb').read()
#     image_dat = tf.image.decode_jpeg(image_raw_data_jpg)
#     print('image_dat1', sess.run(image_dat))
#     image_data = fun(image_dat, sess)
#     print(image_data)
#     plt.imshow(image_data, cmap='gray')
#     plt.show()

# import scipy.misc as misc
# ex_img = tf.gfile.FastGFile('female.jpg', 'rb').read()
# image_dat_tf = tf.image.decode_jpeg(ex_img)
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     image_np = image_dat_tf.eval(session=sess)
# sess = tf.Session()
# image_np = image_dat_tf.eval(session=sess)
# sess.run(tf.global_variables_initializer())
# ex_img_map = gen_stroke(image_np, 8, 250, 200)
# plt.imshow(ex_img_map, cmap='gray')
# plt.show()
# plt.imsave('sketch.jpg', ex_img_map)
# # cv2.imshow('sketch1.jpg', ex_img_map*255.0)
# # cv2.waitKey(0)
# # misc.imsave('sketch2.jpg', ex_img_map*255.0)
