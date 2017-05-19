## 1 运行
```shell

python genplate_plate.py 
```

>执行完毕会在result_dataset下生成一个标注文件test.txt和一个100个车牌图片目录images
> 参见：https://github.com/szad670401/end-to-end-for-chinese-plate-recognition

## 2 修改

>一般情况下，只需要修改最后一行代码：

```python

# 100为数量，result_dataset是输出目录。其它参数不变
G.genBatch(100, 2, range(31, 65), "./result_dataset/", (272, 72))
```

