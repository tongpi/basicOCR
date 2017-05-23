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
Extract background images from a tar archive.

"""


__all__ = (
    'extract_backgrounds',
)


import os
import sys
import tarfile

import cv2
import numpy

BGS_DIR = './bgs'


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def extract_backgrounds(archive_name):
    """
    Extract backgrounds from provided tar archive.

    JPEGs from the archive are converted into grayscale, and cropped/resized to
    256x256, and saved in ./bgs/.

    :param archive_name:
        Name of the .tar file containing JPEGs of background images.

    """
    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:   # 调整背景图片为方形
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:           # 限制背景图片大小<=256*256
            im = cv2.resize(im, (256, 256))
        fname = "{:08}.jpg".format(index)   # 格式化背景图片名：8位整数(图片序号).jpg
        print fname
        rc = cv2.imwrite(BGS_DIR+os.sep+fname, im)  # 保存
        if not rc:
            raise Exception("Failed to write file {}".format(fname))
        index += 1


if __name__ == "__main__":

    if not os.path.exists(BGS_DIR):
        os.mkdir(BGS_DIR)
    
    extract_backgrounds(sys.argv[1])    # 从指定的sys.argv[1]数据压缩包里提取背景图片

