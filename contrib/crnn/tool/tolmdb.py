import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import re
import Image
import numpy as np
import imghdr


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False		
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
			
#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    #print str(len(imagePathList)),str(len(labelList))
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    #print type(env)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):   
        imagePath = './recognition/'+''.join(imagePathList[i]).split()[0].replace('\n','').replace('\r\n','')
        #print imagePath
     
        label = ''.join(labelList[i])
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue	
        #if not imghdr.what(imagePath):
        #    print 'the file is not a pic'
        #    continue
			
        #imgformat = os.path.splitext(imagePath).lower()
        #if not imgformat == '.jpg' or imgformat == '.jpeg' or imgformat == '.png':
        #    print('%s is not a picture' % imagePath)  
        #    continue			
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        #imageBin = cv2.imread(imagePath,0)
        #print(imageBin)
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        #imageBin = rgb2gray(imageBin)
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print cnt
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
	

if __name__ == '__main__':
    outputPath = "./train_lmdb"
    imgdata = open("./train_241.txt")
    imagePathList = list(imgdata)
    
    labelList = []
    for line in imagePathList:
        word = line.split()[1]
        #print word
        labelList.append(word)
    createDataset(outputPath, imagePathList, labelList)
    #pass
