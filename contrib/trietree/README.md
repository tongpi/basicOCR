# trietree
该代码是基于字典树对word的识别结果进行矫正，使用于中英文混合的字典。字典树（trietree）：常用应用于大量字符串的保存、统计、查找等操作。

##  src：矫正word识别结果
   trietree_correct.py是主要代码文件；  
   矫正word识别结果函数：correct_word("复合", 1, trieTree.trie)  
   第一个参数是待矫正word；  
   第二个参数是编辑距离，一般取3，包含3；  
   第三个参数是根据字典txt文件构建的字典树。     
   
   dict.txt等txt文件是含有汉字、英文的字典；每行包含词、词频，用空格隔开；  
   test.py是测试文件。
  

## wordFrequency：统计词频
   stopword_path = r'stopwords.dat'  ：停词，每行存放一个忽略的词，可以是标点符号等。  
   inputpath = r'words.txt'   : 输入，格式是分过词的，每个词用空格分开。  
   outputpath = r'dict_new.txt' ：输出，格式是每行词、词频，用空格隔开，也就是trietree_correct.py需要的字典。
   
 # 参考文献：
 trietree: http://stevehanov.ca/blog/index.php?id=114   
 中英文统一编码：  http://blog.csdn.net/qinbaby/article/details/23201883
