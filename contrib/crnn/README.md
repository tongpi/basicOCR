fork from meijieru/crnn.pytorch https://github.com/meijieru/crnn.pytorch
## crnn实现细节(pytorch)
### 1.环境搭建
#### 1.1 基础环境
* Ubuntu14.04 + CUDA
* opencv2.4 + pytorch + lmdb +wrap_ctc

安装lmdb `apt-get install lmdb`
#### 1.2 安装pytorch
pip,linux,cuda8.0,python2.7:pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
参考：http://pytorch.org/
#### 1.3 安装wrap_ctc
    git clone https://github.com/baidu-research/warp-ctc.git`
    cd warp-ctc
    mkdir build; cd build
    cmake ..
    make

GPU版在环境变量中添加
    export CUDA_HOME="/usr/local/cuda"

    cd pytorch_binding
    python setup.py install
    
参考：https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding
#### 1.4 注意问题
1. 缺少cffi库文件 使用`pip install cffi`安装
2. 安装pytorch_binding前,确认设置CUDA_HOME,虽然编译安装不会报错,但是在调用gpu时，会出现wrap_ctc没有gpu的属性的错误
### 2. crnn预测
运行`/contrib/crnn/demo.py`

原始图片为: ![](./media/image31.png)

识别结果为：A-----v--a-i-l-a-bb-l-ee-- => Available
    
    # 加载模型
    model_path = './samples/netCRNN_9_112580.pth'
    # 需识别的图片
    img_path = './data/demo.png'
    # 识别的类别
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # 设置模型参数 图片高度imgH=32, nc, 分类数目nclass=len(alphabet)+1 一个预留位, LSTM设置隐藏层数nh=256, 使用GPU个数ngpu=1
    model = crnn.CRNN(32, 1, 63, 256, 1).cuda()

替换模型时，注意模型分类的类别数目
## crnn 训练
1. 数据预处理

运行`/contrib/crnn/tool/tolmdb.py`

    # 生成的lmdb输出路径
    outputPath = "./train_lmdb"
    # 图片及对应的label
    imgdata = open("./train.txt")

2. 训练模型

运行`/contrib/crnn/crnn_main.py`

    python crnn_main.py [--param val]
    --trainroot        训练集路径
    --valroot          验证集路径
    --workers          CPU工作核数, default=2
    --batchSize        设置batchSize大小, default=64
    --imgH             图片高度, default=32
    --nh               LSTM隐藏层数, default=256
    --niter            训练回合数, default=25
    --lr               学习率, default=0.01
    --beta1             
    --cuda             使用GPU, action='store_true'
    --ngpu             使用GPU的个数, default=1
    --crnn             选择预训练模型
    --alphabet         设置分类
    --Diters            
    --experiment        模型保存目录
    --displayInterval   设置多少次迭代显示一次, default=500
    --n_test_disp        每次验证显示的个数, default=10
    --valInterval        设置多少次迭代验证一次, default=500
    --saveInterval       设置多少次迭代保存一次模型, default=500
    --adam               使用adma优化器, action='store_true'
    --adadelta           使用adadelta优化器, action='store_true'
    --keep_ratio         设置图片保持横纵比缩放, action='store_true'
    --random_sample      是否使用随机采样器对数据集进行采样, action='store_true'
    
    

