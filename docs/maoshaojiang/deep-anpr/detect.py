#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
    'detect',
    'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model


def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5       # 设定缩放比例
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))  # 缩放图片
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:  # 若缩放后图片尺寸满足该条件,停止缩放迭代
            break
        yield cv2.resize(im, (shape[1], shape[0]))  # 返回缩放后的图片


def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))      # 针对需检测图片产生迭代缩放图片列表

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()     # 获取检测网络(5个卷积层)的输入,输出以及所有参数

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:   # 建立会话
        y_vals = []
        for scaled_im in scaled_ims:        # 遍历缩放图片列表
            feed_dict = {x: numpy.stack([scaled_im])}   # 建立字典feed_dict{输入：缩放图片}
            feed_dict.update(dict(zip(params, param_vals))) # 将字典{参数：参数值}更新到字典feed_dict中->{输入：缩放图片, 参数：参数值}
            y_vals.append(sess.run(y, feed_dict=feed_dict)) # 提供检测图片和网络参数值,执行检测操作

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):    # 枚举元组(缩放图片, 检测输出)及对应下标
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)): # 对应sigmoid输出号码牌存在概率大于0.99的下标
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)     # 车牌号识别正确的概率分布

            img_scale = float(im.shape[0]) / scaled_im.shape[0] # 缩放比例

            bbox_tl = window_coords * (8, 4) * img_scale        # 计算缩放图片上号码牌的boudingbox的左上角坐标
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale     # 号码牌的boudingbox的尺寸

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0]) # 号码牌是否存在的概率分布

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs  # 返回号码牌boudingbox的左上角坐标,右下角坐标,
                                                                            #     号码牌是否存在的概率分布,车牌号识别正确的概率分布


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])      # 若两个boudingbox重叠则返回True,否则返回False


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {0:0}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
            else:
                match_to_group[idx1] = num_groups 
                num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.

    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.

    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1)) # 返回识别出来的车牌号


if __name__ == "__main__":
    
    im = cv2.imread(sys.argv[1])        # 读取需检测的图片
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.       # 灰度化,归一化

    f = numpy.load(sys.argv[2])     # 加载训练好的模型参数值文件
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    for pt1, pt2, present_prob, letter_probs in post_process(
                                                  detect(im_gray, param_vals)): # 获取号码牌boudingbox的左上角坐标,右下角坐标,
                                                                                #     号码牌是否存在的概率分布,车牌号识别正确的概率分布
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))    # 坐标值整型化并从(y, x)颠倒为(x, y)

        code = letter_probs_to_code(letter_probs)   # 获取识别出来的车牌号

        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)  # 在原检测图片上绘制一个绿色的boudingbox

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 0),
                    thickness=5)        # 在boudingbox的左上角绘制识别出来的车牌号字符串(黑色)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)        # 在boudingbox的左上角绘制识别出来的车牌号字符串(白色)

    cv2.imwrite(sys.argv[3], im)    # 保存该检测图片到sys.argv[3]

