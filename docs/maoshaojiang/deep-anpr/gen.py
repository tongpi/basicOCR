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
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

SYNDATA_DIR = './syndata'

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4       # 设定字体尺寸

    font = ImageFont.truetype(font_path, font_size)     # 创建给定字体文件和尺寸的字体对象

    height = max(font.getsize(c)[1] for c in CHARS)     # 获取字符集中所有字符的最大高度(包括空格字符)

    for c in CHARS:
        width = font.getsize(c)[0]       # 获取指定字符的宽度
        im = Image.new("RGBA", (width, height), (0, 0, 0))      # 创建指定模式、大小和颜色的图片对象

        draw = ImageDraw.Draw(im)       # 创建指定图片对象的绘制对象
        draw.text((0, 0), c, (255, 255, 255), font=font)        # 以指定位置、颜色和字体对象绘制该字符
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)    # 调整该字符图片大小,并做抗锯齿处理
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.      # 返回该字符及单通道归一化的字符图片


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M        # 返回一个经过三维旋转的3*3矩阵


def pick_colors():
    first = True
    
    while first or plate_color - text_color < 0.3:      # 选择文本颜色和牌子颜色,使牌子颜色-文本颜色>=0.3
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color      # 返回文本颜色和牌子颜色


def make_affine_transform(from_shape, to_shape,         # from_shape=plate.shape, to_shape=bg.shape,
                          min_scale, max_scale,         # min_scale=0.6, max_scale=0.875,
                          scale_variation=1.0,          # scale_variation=1.5,
                          rotation_variation=1.0,       # rotation_variation=1.0,
                          translation_variation=1.0):   # translation_variation=1.2
    out_of_bounds = False       # 车牌存在标志位

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T     # 
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:      # 车牌尺寸过大或过小都认为不存在
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation       # 绕z轴旋转角度
    pitch = random.uniform(-0.2, 0.2) * rotation_variation      # 绕x轴旋转角度
    yaw = random.uniform(-1.2, 1.2) * rotation_variation        # 绕y轴旋转角度

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]      # 从经过三维旋转的3*3矩阵提取其左上角的2*2矩阵
    h, w = from_shape       # 号码牌的高和宽
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):   # 车牌不完整认为不存在
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    return "{}{}{}{} {}{}{}".format(
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS))  # 返回一个8位字符串：'字母字母数字数字 字母字母字母'(包括1个空格)


def rounded_rect(shape, radius):
    out = numpy.ones(shape)     # 初始化一个号码牌大小的全1数组
    out[:radius, :radius] = 0.0     # 左上角取0.0
    out[-radius:, :radius] = 0.0    # 左下角取0.0
    out[:radius, -radius:] = 0.0    # 右上角取0.0
    out[-radius:, -radius:] = 0.0   # 右下角角取0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)      # 左上角画圆
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)       # 左下角画圆
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)       # 右上角画圆
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)        # 右下角画圆

    return out      # 返回一个圆角化的矩形数组


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height      # h_padding=[6.4, 12.8]中的一个随机浮点数,水平填充
    v_padding = random.uniform(0.1, 0.3) * font_height      # v_padding=[3.2, 9.6]中的一个随机浮点数,垂直填充
    spacing = font_height * random.uniform(-0.05, 0.05)     # spacing=[-1.6, 1.6]中的一个随机浮点数,字符间隔
    radius = 1 + int(font_height * 0.1 * random.random())   # radius=[1, 4)中的一个整数,矩形圆角半径

    code = generate_code()      # 生成7位车牌号：字母字母数字数字 字母字母字母(包括1个空格)
    text_width = sum(char_ims[c].shape[1] for c in code)    # 对7位车牌号字符宽度求和(包括1个空格)
    text_width += (len(code) - 1) * spacing     # 加上[-11.2, 11.2]中的一个随机浮点数,文本宽度

    out_shape = (int(font_height + v_padding * 2), int(text_width + h_padding * 2))   # 号码牌尺寸

    text_color, plate_color = pick_colors()     # 选择满足条件(牌子颜色-文本颜色>=0.3)的文本颜色和牌子颜色
    
    text_mask = numpy.zeros(out_shape)      # 初始化号码牌数组
    
    x = h_padding
    y = v_padding 
    for c in code:      # 将文本填充进初始号码牌数组
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) + numpy.ones(out_shape) * text_color * text_mask)   # 产生号码牌

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")    # 返回号码牌,号码牌大小的圆角矩形以及7位车牌号(没有空格)


def generate_bg(num_bg_images):
    found = False
    
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))   # 随机抽取一张背景图片
        bg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.      # 读取该背景图片,并灰度化和归一化
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]       # 如果该背景图片满足：宽>=128且高>=64,
                                                                # 则从该背景图片中随机抠取一张高*宽=64*128的图片

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)     # 生成一张高*宽=64*128的背景图片

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)     # 生成号码牌,号码牌大小的圆角矩形以及7位车牌号(没有空格)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)      # 合成带背景的号码牌图片

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))   # 调整图片大小为128*64

    out += numpy.random.normal(scale=0.05, size=out.shape)      # 加入高斯噪音
    out = numpy.clip(out, 0., 1.)       # 限制合成图片的像素值在[0, 1]之间

    return out, code, not out_of_bounds     # 返回合成图片,7位车牌号以及取反的存在标志位(0代表不存在,1否)


def load_fonts(folder_path):
    font_char_ims = {}      # 初始化一个字典
    
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]  # 生成.ttf格式的字体文件列表
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path, font), FONT_HEIGHT)) # 迭代生成一个字典{字符：字符图片},
                                                                                                # 并将其作为字典font_char_ims的font键对应的value值
    
    return fonts, font_char_ims     # 返回该字体文件列表和字典{font：{字符：字符图片}}


def generate_ims():
    """
    Generate number plate images.
    return:Iterable of number plate images.
    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)     # 加载字体文件列表和字典{font：{字符：字符图片}}
    num_bg_images = len(os.listdir("bgs"))      # 获取背景图片的数量
    
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)


if __name__ == "__main__":
        
    if not os.path.exists(SYNDATA_DIR):        # 创建合成图片目录
        os.mkdir(SYNDATA_DIR)
    
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))     # 以generate_ims()迭代对象创建一个迭代器,
                                                                    # 迭代生成int(sys.argv[1])个号码牌图片
    
    for img_idx, (im, c, p) in enumerate(im_gen):       # 枚举出每张号码牌图片(图片,车牌号,存在标志位)及对应下标
        fname = "{:08d}_{}_{}.png".format(img_idx, c, "1" if p else "0")   # 格式化号码牌图片的文件名：
                                                                           # 8位整数(图片序号)_7位字符(车牌号)_1位标志位(0代表号码牌不存在,1否).png
        print fname
        cv2.imwrite(SYNDATA_DIR+os.sep+fname, im*255.)       # 保存号码牌图片到SYNDATA_DIR目录下
        

