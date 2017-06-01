#coding=utf-8
#!/usr/bin/python
#By Steve Hanov, 2011. Released to the public domain 
import time
import sys
import codecs 
import json  

#step1：中英文混合编码，下面代码包含了判断unicode是否是汉字、数字、英文或者其他字符，全角符号转半角符号，unicode字符串归一化等工作。
#       中英文混合字串的统一编码表示中英文混合字串处理最省力的办法就是把它们的编码都转成 Unicode，让一个汉字与一个英文字母的内存位宽都是相等的。
#       http://blog.csdn.net/qinbaby/article/details/23201883
#step2: 2.1先根据字典txt文件构建字典树；2.2其次寻找编辑距离小于等于某个数值的所有相似的word；2.3最后对一堆相似word后处理，决定最终的、经过矫正的word. 
#       http://stevehanov.ca/blog/index.php?id=114   字典txt的格式:每行 (word  word frequency) 

#备注：
#1.先构建字典树，只操作一次；其次使用correct_word()矫正识别结果。
#2. 采样utf-8编码（解决国际上字符的一种多字节编码），它对英文使用8位（即一个字节），中文使用24为（三个字节）来编码。
#  由于英文、汉字各占的字节数不同，所以先采用uniform(ustring)进行归一化处理（统一编码）。

#step1:
def is_chinese(uchar): #判断一个unicode是否是汉字 
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_alphabet(uchar):#判断一个unicode是否是英文字母"""    
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False

def is_number(uchar):#判断一个unicode是否是数字"""  
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False
                
def is_other(uchar):#判断是否非汉字，数字和英文字符"""   
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False
                
def B2Q(uchar):#半角转全角"""   
    inside_code=ord(uchar)
    if inside_code<0x0020 or inside_code>0x7e:      #不是半角字符就返回原来的字符
        return uchar
    if inside_code==0x0020: #除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code=0x3000
    else:
        inside_code+=0xfee0

    return unichr(inside_code)
        
def Q2B(uchar):#全角转半角"""    
    inside_code=ord(uchar)
    if inside_code==0x3000:
        inside_code=0x0020
    else:
        inside_code-=0xfee0

    if inside_code<0x0020 or inside_code>0x7e:      #转完之后不是半角字符返回原来的字符
        return uchar
        
    return unichr(inside_code)

def stringQ2B(ustring): #把字符串全角转半角"""   
    return "".join([Q2B(uchar) for uchar in ustring])

def uniform(ustring):#格式化字符串，完成全角转半角，大写转小写的工作"""   
    return stringQ2B(ustring).lower()

def string2List(ustring):#将ustring按照中文，字母，数字分开"""   
    retList=[]
    utmp=[]

    for uchar in ustring:
        if is_other(uchar):
            if len(utmp)==0:
                continue
            else:
                retList.append("".join(utmp))
                utmp=[]
        else:
            utmp.append(uchar)

        if len(utmp)!=0:
            retList.append("".join(utmp))

    return retList
'''  #例子
if __name__=="__main__":

    ustring=u'我的English学的不好'
    ustring=uniform(ustring)
    print len(ustring)
    print ustring[4]
    ret=string2List(ustring)
    print ret'''


#step2:
#step2.1：
# Keep some interesting statistics
NodeCount = 0
        
# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
class TrieNode:
   
    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert( self, word, freq):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word
        node.freq = freq
        
        
    
def construction_trietree(DICTIONARY):  
    WordCount = 0 
    # read dictionary file into a trie
    trie = TrieNode()
    ann_file = codecs.open(DICTIONARY, 'r', 'utf-8') #中文使用gbk 或者 utf-8#open(DICTIONARY, "rt").read().split():
    lines = ann_file.readlines()
    for l in lines:
        l=uniform(l)
        word, word_freq = l.strip().split()               
        WordCount += 1
        trie.insert( word,  word_freq)
        #print word, word_fre   
    ann_file.close()
    print "Read %d words into %d nodes" % (WordCount, NodeCount)

    return trie

#step2.2：
# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search( word, maxCost, trie):
    # build first row
    currentRow = range( len(word) + 1 )

    results = []

    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive( trie.children[letter], letter, word, currentRow, 
            results, maxCost )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already.
def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in xrange( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1
        #print (letter)
        #print (len(word)) 
        #print (word[0])
        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:                
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word != None:
        results.append( (node.word, currentRow[-1], node.freq ) )

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min( currentRow ) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow, 
                results, maxCost )
                
#step2.3：               
def postProcessing(TARGET, words, distance, freq):
    '''words = []
    distance = []
    
    for l in lst:
        s = str(l)
        s = s[1:len(s)-1]
        word, dis = s.strip().split(", ") # Levenshtein编辑距离
        words.append(word[1:len(word)-1])
        distance.append(dis)
        print word, dis'''
    
    dis_min = min(distance)
    #print ("min:{}\n".format(dis_min))
    #print ("min index:{}\n".format(distance.index(dis_min)))
    words_min = []
    words_min_freq = []
    for i in range(len(words)):
        if distance[i]==dis_min:
            words_min.append( words[i] )
            words_min_freq.append( freq[i] )
    #print words_min
    
    if int(dis_min)==0: #最小编辑距离是0，表示是单词被识别正确 
        word = words[distance.index(dis_min)]
        #print word
        return word
    else: #最小编辑距离不是0，表示识别有误
        if len(words_min)==1: #表示只有一个最接近目标word
            return words[distance.index(dis_min)]
        else: 
            for i in range(len(words_min)):
                #print words_min[i]
                #print len(TARGET), len(words_min[i])
                if len(TARGET)==len(words_min[i]):
                    return words_min[i]
                else: #输出词频最大的那个word 
                    '''print (max(words_min_freq))
                    print (words_min[0])
                    print ( words_min_freq.index( max(words_min_freq) ))'''
                    return  words_min[words_min_freq.index( max(words_min_freq) )]
                
                
def correct_word(TARGET, MAX_COST, trie): #MAX_COST 一般取3
    start = time.time()
    #print (len(TARGET))
    TARGET = str(TARGET).decode("utf-8") #unicode(TARGET,"utf-8") #str转化为unicode可以通过unicode()
    TARGET = uniform(TARGET)
    #print (len(TARGET))
    words = []
    dis = []
    freq = []
    
    results = search( TARGET, MAX_COST, trie) 
    if len(results)<1:
        results = search( TARGET, MAX_COST+3, trie)
        if len(results)<1: #如果字典里面实在没有该word就不需要矫正     
            return TARGET
        
    for result in results: #u'\u9519\u4f4d'
        words.append(result[0].encode("utf-8"))
        dis.append(result[1])
        freq.append(float(result[2])) #词频
        #print result[0].encode("utf-8"),result[1], result[2]
        
    end = time.time()

    #for result in results: print result        
    word = postProcessing(TARGET, words, dis, freq) #后处理
    #print ("word:{}\n".format(word))
    
    print "Search took %g s" % (end - start)
    return word

    