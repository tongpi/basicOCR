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
Routines for training the network.

"""


__all__ = (
    'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model


def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])        # 令7位车牌号的每一位在36个字符中的对应字符位为1

    return numpy.concatenate([[1. if p else 0], c.flatten()])   # 展平,再衔接1位出界标志位返回1+7*36个元素的列表


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1][9:16]        # 7位车牌号
        p = fname.split("/")[1][17] == '1'      # 1位出界标志位
        yield im, code_to_vec(p, code)      # 返回合成图片,1+7*36个元素的列表


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys       # 返回图片,标签数组集


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped
        

@mpgen
def read_batches(batch_size):
    g = gen.generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())


def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 1:],
                                                     [-1, len(common.CHARS)]),  # [bs, 7*36]->[bs*7, 36]
                                          tf.reshape(y_[:, 1:],
                                                     [-1, len(common.CHARS)]))  # [n, 7*36]->[n*7, 36]
    digits_loss = tf.reshape(digits_loss, [-1, 7])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    digits_loss *= (y_[:, 0] != 0)
    digits_loss = tf.reduce_sum(digits_loss)        # 7位车牌号(车牌存在的情况)的每一位对应36个字符的概率分布的多元交叉熵

    # Calculate the loss from presence indicator being wrong.
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                                          y[:, :1], y_[:, :1])
    presence_loss = 7 * tf.reduce_sum(presence_loss)        # 车牌是否存在的概率分布的二元交叉熵

    return digits_loss, presence_loss, digits_loss + presence_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    x, y, params = model.get_training_model()       # 获取训练网络(3个卷积层+2个全连接层)的输入,输出以及所有参数

    y_ = tf.placeholder(tf.float32, [None, 1 + 7 * len(common.CHARS)])  # 给定网络输入对应的真输出[n, 1+7*36]

    digits_loss, presence_loss, loss = get_loss(y, y_)      # 获取损失函数
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)      # 采用Adam优化器对损失函数进行优化

    best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)       # 7位车牌号的每一位对应概率最大的字符的下标
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(common.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()        # 初始化所有变量

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best,     # 提供测试数据集,执行测试操作
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      loss],
                     feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(        # 车牌存在的情况下,车牌号识别正确的个数
                        numpy.logical_or(
                            numpy.all(r[0] == r[1], axis=1),
                            numpy.logical_and(r[2] < 0.5,
                                              r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(vec_to_plate(c), pc,         # 输出：给定的7位车牌号 给定的车牌存在与否标志位
                                           vec_to_plate(b), float(pb))  #       <-> 测试得到的7位车牌号 测试得到的车牌存在概率
        num_p_correct = numpy.sum(r[2] == r[3])     # 车牌存在与否判断正确的个数

        print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "      # 输出：B第n批次 车牌号识别准确率 车牌存在与否判断准确率 loss：总偏差
               "(digits: {}, presence: {}) |{}|").format(   #       (digits：车牌号识别偏差, presence：车牌存在与否判断偏差)
            batch_idx,                                      #       |batch_size个2类字符| 识别错误显示字符'X',识别正确显示字符' '
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            r[4],
            r[5],
            "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                                           for b, c, pb, pc in zip(*r_short)))

    def do_batch():
        sess.run(train_step,        # 批量提供训练数据集,执行训练操作
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:       # 训练1个回合进行1次测试，并显示相关指标
            do_report()     # 显示训练效果

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    # 配置GPU资源,建立会话
        #with tf.device("/gpu:1"):
        sess.run(init)      # 执行变量初始化
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("syndata/*.png"))[:50])     # 取前50个样本作为测试集

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))        # 枚举出batch_size个训练数据集和对应下标
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()      # 批处理
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(      # 计算训练60批次(1批batch_size张图片)的用时
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:       # 出现键盘中断异常,则保存此刻的模型参数值到weights.npz文件
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights


if __name__ == "__main__":
    
    if len(sys.argv) > 1:           # 加载训练好的权重文件
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None      # 如果没有训练好的权重，就令初始权重为None

    train(learn_rate=0.001,                 # 学习率为0.001
          report_steps=20,                  # 报告步数为20（即1个回合显示一次日志）
          batch_size=50,                    # 批尺寸为50
          initial_weights=initial_weights)  # 初始权重为提供的.npz文件或None

