# -*- coding:utf-8 -*- 

import trietree_correct as trieTree

#加载字典
trieTree.trie = trieTree.construction_trietree("dict.txt") #("userdic.txt")

#输出矫正后的word
word = trieTree.correct_word("复合", 2, trieTree.trie)
print (word)



 
 
