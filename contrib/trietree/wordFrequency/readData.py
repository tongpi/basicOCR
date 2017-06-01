#coding=utf-8
import jieba
from collections import Counter
import codecs 

stopword_path = r'stopwords.dat'
class seg_word(object):
	def __init__(self, inputpath, outputpath):
		self.inputpath = inputpath
		self.outputpath = outputpath

	def cut_data(self):
		with codecs.open(inputpath, 'r', 'utf-8') as fr:
			res = jieba.cut(fr.read())
		return res

	def output_file(self):
		outcome = self.cumpute_word_count()
		with codecs.open(outputpath, 'w', 'utf-8') as fw:
			for k,v in outcome.items():
				fw.write(k + ' ' + str(v) + '\n')

	def filter_data(self):
		with codecs.open(stopword_path,'r','utf-8') as f:
			stopword_list = f.read()
		seg_res = self.cut_data()
		res_list = [word.strip() for word in seg_res if word not in stopword_list]
		return res_list

	def cumpute_word_count(self):
		res_word = self.filter_data()
		word_freq = dict(Counter(res_word))
		return word_freq



if __name__ == '__main__':
	inputpath = r'words.txt'
	outputpath = r'dict_new.txt'
	c = seg_word(inputpath, outputpath)
	res = c.output_file()
