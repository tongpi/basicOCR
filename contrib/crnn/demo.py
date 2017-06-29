#coding: utf-8
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

model_path = './data/netCRNN_ch_nc_21_nh_128.pth'
img_path = './data/image33.jpg'
alphabet = u'\'ACIMRey万下依口哺摄次状璐癌草血运重'
#print(alphabet)
nclass = len(alphabet) + 1
model = crnn.CRNN(32, 1, nclass, 128, 1).cuda()
print('loading pretrained model from %s' % model_path)
pre_model = torch.load(model_path)
for k,v in pre_model.items():
    print(k,len(v))
model.load_state_dict(pre_model)

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image).cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.squeeze(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred.encode('utf8'), sim_pred.encode('utf8')))
