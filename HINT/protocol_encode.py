'''
input:
	data/raw_data.csv

output:
	data/sentence2embedding.pkl (preprocessing)
	protocol_embedding 
'''

import csv, pickle 
from functools import reduce
from tqdm import tqdm 
import torch 
torch.manual_seed(0)
from torch import nn 
import torch.nn.functional as F

def clean_protocol(protocol):
	protocol = protocol.lower()
	protocol_split = protocol.split('\n')
	filter_out_empty_fn = lambda x: len(x.strip())>0
	strip_fn = lambda x:x.strip()
	protocol_split = list(filter(filter_out_empty_fn, protocol_split))	
	protocol_split = list(map(strip_fn, protocol_split))
	return protocol_split 

def get_all_protocols():
	input_file = 'data/raw_data.csv'
	with open(input_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	protocols = [row[9] for row in rows]
	return protocols

def split_protocol(protocol):
	protocol_split = clean_protocol(protocol)
	inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)	
	for idx, sentence in enumerate(protocol_split):
		if "inclusion" in sentence:
			inclusion_idx = idx
			break
	for idx, sentence in enumerate(protocol_split):
		if "exclusion" in sentence:
			exclusion_idx = idx 
			break 		
	if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
		inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
		exclusion_criteria = protocol_split[exclusion_idx:]
		if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
			print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
			exit()
		return inclusion_criteria, exclusion_criteria ## list, list 
	else:
		return protocol_split, 

def collect_cleaned_sentence_set():
	protocol_lst = get_all_protocols() 
	cleaned_sentence_lst = []
	for protocol in protocol_lst:
		result = split_protocol(protocol)
		cleaned_sentence_lst.extend(result[0])
		if len(result)==2:
			cleaned_sentence_lst.extend(result[1])
	return set(cleaned_sentence_lst)


def save_sentence_bert_dict_pkl():
	cleaned_sentence_set = collect_cleaned_sentence_set() 
	from biobert_embedding.embedding import BiobertEmbedding
	biobert = BiobertEmbedding()
	def text2vec(text):
		return biobert.sentence_vector(text)
	protocol_sentence_2_embedding = dict()
	for sentence in tqdm(cleaned_sentence_set):
		protocol_sentence_2_embedding[sentence] = text2vec(sentence)
	pickle.dump(protocol_sentence_2_embedding, open('data/sentence2embedding.pkl', 'wb'))
	return 

def load_sentence_2_vec():
	sentence_2_vec = pickle.load(open('data/sentence2embedding.pkl', 'rb'))
	return sentence_2_vec 

def protocol2feature(protocol, sentence_2_vec):
	result = split_protocol(protocol)
	inclusion_criteria, exclusion_criteria = result[0], result[-1]
	inclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in inclusion_criteria if sentence in sentence_2_vec]
	exclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in exclusion_criteria if sentence in sentence_2_vec]
	if inclusion_feature == []:
		inclusion_feature = torch.zeros(1,768)
	else:
		inclusion_feature = torch.cat(inclusion_feature, 0)
	if exclusion_feature == []:
		exclusion_feature = torch.zeros(1,768)
	else:
		exclusion_feature = torch.cat(exclusion_feature, 0)
	return inclusion_feature, exclusion_feature 


class Protocol_Embedding(nn.Sequential):
	def __init__(self, output_dim, highway_num, device ):
		super(Protocol_Embedding, self).__init__()	
		self.input_dim = 768  
		self.output_dim = output_dim 
		self.highway_num = highway_num 
		self.fc = nn.Linear(self.input_dim*2, output_dim)
		self.f = F.relu
		self.device = device 
		self = self.to(device)

	def forward_single(self, inclusion_feature, exclusion_feature):
		## inclusion_feature, exclusion_feature: xxx,768 
		inclusion_feature = inclusion_feature.to(self.device)
		exclusion_feature = exclusion_feature.to(self.device)
		inclusion_vec = torch.mean(inclusion_feature, 0)
		inclusion_vec = inclusion_vec.view(1,-1)
		exclusion_vec = torch.mean(exclusion_feature, 0)
		exclusion_vec = exclusion_vec.view(1,-1)
		return inclusion_vec, exclusion_vec 

	def forward(self, in_ex_feature):
		result = [self.forward_single(in_mat, ex_mat) for in_mat, ex_mat in in_ex_feature]
		inclusion_mat = [in_vec for in_vec, ex_vec in result]
		inclusion_mat = torch.cat(inclusion_mat, 0)  #### 32,768
		exclusion_mat = [ex_vec for in_vec, ex_vec in result]
		exclusion_mat = torch.cat(exclusion_mat, 0)  #### 32,768 
		protocol_mat = torch.cat([inclusion_mat, exclusion_mat], 1)
		output = self.f(self.fc(protocol_mat))
		return output 

	@property
	def embedding_size(self):
		return self.output_dim 



if __name__ == "__main__":
	# protocols = get_all_protocols()
	# split_protocols(protocols)
	save_sentence_bert_dict_pkl() 











