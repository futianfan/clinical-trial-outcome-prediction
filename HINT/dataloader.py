'''

(I). Trial_Dataset for prediction
(II). Trial_Dataset_Complete for interpretation
(III). SMILES lst 
(IV). disease lst icd-code 

'''

import torch, csv, os
from torch.utils import data 
from torch.utils.data.dataloader import default_collate
from HINT.molecule_encode import smiles2mpnnfeature
from HINT.protocol_encode import protocol2feature, load_sentence_2_vec

sentence2vec = load_sentence_2_vec() 

class Trial_Dataset(data.Dataset):
	def __init__(self, nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst):
		self.nctid_lst = nctid_lst 
		self.label_lst = label_lst 
		self.smiles_lst = smiles_lst 
		self.icdcode_lst = icdcode_lst 
		self.criteria_lst = criteria_lst 

	def __len__(self):
		return len(self.nctid_lst)

	def __getitem__(self, index):
		return self.nctid_lst[index], self.label_lst[index], self.smiles_lst[index], self.icdcode_lst[index], self.criteria_lst[index]
	#### smiles_lst[index] is list of smiles

class Trial_Dataset_Complete(Trial_Dataset):
	def __init__(self, nctid_lst, status_lst, why_stop_lst, label_lst, phase_lst, 
					   diseases_lst, icdcode_lst, drugs_lst, smiles_lst, criteria_lst):
		Trial_Dataset.__init__(self, nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst)
		self.status_lst = status_lst 
		self.why_stop_lst = why_stop_lst 
		self.phase_lst = phase_lst 
		self.diseases_lst = diseases_lst 
		self.drugs_lst = drugs_lst 

	def __getitem__(self, index):
		return self.nctid_lst[index], self.status_lst[index], self.why_stop_lst[index], self.label_lst[index], self.phase_lst[index], \
			   self.diseases_lst[index], self.icdcode_lst[index], self.drugs_lst[index], self.smiles_lst[index], self.criteria_lst[index]



class ADMET_Dataset(data.Dataset):
	def __init__(self, smiles_lst, label_lst):
		self.smiles_lst = smiles_lst 
		self.label_lst = label_lst 
	
	def __len__(self):
		return len(self.smiles_lst)

	def __getitem__(self, index):
		return self.smiles_lst[index], self.label_lst[index]

def admet_collate_fn(x):
	smiles_lst = [i[0] for i in x]
	label_vec = default_collate([int(i[1]) for i in x])  ### shape n, 
	return [smiles_lst, label_vec]


def smiles_txt_to_lst(text):
	"""
		"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
	"""
	text = text[1:-1]
	lst = [i.strip()[1:-1] for i in text.split(',')]
	return lst 

def icdcode_text_2_lst_of_lst(text):
	text = text[2:-2]
	lst_lst = []
	for i in text.split('", "'):
		i = i[1:-1]
		lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
	return lst_lst 

def trial_collate_fn(x):
	nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
	label_vec = default_collate([int(i[1]) for i in x])  ### shape n, 
	smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
	icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
	criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
	return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst]

def trial_complete_collate_fn(x):
	nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
	status_lst = [i[1] for i in x]
	why_stop_lst = [i[2] for i in x]
	label_vec = default_collate([int(i[3]) for i in x])  ### shape n, 
	phase_lst = [i[4] for i in x]
	diseases_lst = [i[5] for i in x]
	icdcode_lst = [icdcode_text_2_lst_of_lst(i[6]) for i in x]
	drugs_lst = [i[7] for i in x]
	smiles_lst = [smiles_txt_to_lst(i[8]) for i in x]
	criteria_lst = [protocol2feature(i[9], sentence2vec) for i in x]
	return [nctid_lst, status_lst, why_stop_lst, label_vec, phase_lst, diseases_lst, icdcode_lst, drugs_lst, smiles_lst, criteria_lst]

def csv_three_feature_2_dataloader(csvfile, shuffle, batch_size):
	with open(csvfile, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))[1:]
	## nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria
	nctid_lst = [row[0] for row in rows]
	label_lst = [row[3] for row in rows]
	icdcode_lst = [row[6] for row in rows]
	drugs_lst = [row[7] for row in rows]
	smiles_lst = [row[8] for row in rows]
	criteria_lst = [row[9] for row in rows] 
	dataset = Trial_Dataset(nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst)
	data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = trial_collate_fn)
	return data_loader

def csv_three_feature_2_complete_dataloader(csvfile, shuffle, batch_size):
	with open(csvfile, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))[1:]	
	nctid_lst = [row[0] for row in rows]
	status_lst = [row[1] for row in rows]
	why_stop_lst = [row[2] for row in rows]
	label_lst = [row[3] for row in rows]
	phase_lst = [row[4] for row in rows]
	diseases_lst = [row[5] for row in rows]
	icdcode_lst = [row[6] for row in rows]
	drugs_lst = [row[7] for row in rows]
	smiles_lst = [row[8] for row in rows]
	new_drugs_lst, new_smiles_lst = [], []
	criteria_lst = [row[9] for row in rows] 
	dataset = Trial_Dataset_Complete(nctid_lst, status_lst, why_stop_lst, label_lst, phase_lst, 
					   				 diseases_lst, icdcode_lst, drugs_lst, smiles_lst, criteria_lst)
	data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = trial_complete_collate_fn)
	return data_loader 






def smiles_txt_to_2lst(smiles_txt_file):
	with open(smiles_txt_file, 'r') as fin:
		lines = fin.readlines() 
	smiles_lst = [line.split()[0] for line in lines]
	label_lst = [int(line.split()[1]) for line in lines]
	return smiles_lst, label_lst 

def generate_admet_dataloader_lst(batch_size):
	datafolder = "data/ADMET/cooked/"
	name_lst = ["absorption", 'distribution', 'metabolism', 'excretion', 'toxicity']
	dataloader_lst = []
	for i,name in enumerate(name_lst):
		train_file = os.path.join(datafolder, name + '_train.txt')
		test_file = os.path.join(datafolder, name +'_valid.txt')
		train_smiles_lst, train_label_lst = smiles_txt_to_2lst(train_file)
		test_smiles_lst, test_label_lst = smiles_txt_to_2lst(test_file)
		train_dataset = ADMET_Dataset(smiles_lst = train_smiles_lst, label_lst = train_label_lst)
		test_dataset = ADMET_Dataset(smiles_lst = test_smiles_lst, label_lst = test_label_lst)
		train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
		test_dataloader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
		dataloader_lst.append((train_dataloader, test_dataloader))
	return dataloader_lst 

# ## x is a list, len(x)=batch_size, x[i] is tuple, len(x[0])=5  
# def mpnn_feature_collate_func(x): 
# 	return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]


# def mpnn_collate_func(x):
# 	#print("len(x) is ", len(x)) ## batch_size 
# 	#print("len(x[0]) is ", len(x[0])) ## 3--- data_process_loader.__getitem__ 
# 	mpnn_feature = [i[0] for i in x]
# 	#print("len(mpnn_feature)", len(mpnn_feature), "len(mpnn_feature[0])", len(mpnn_feature[0]))
# 	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
# 	from torch.utils.data.dataloader import default_collate
# 	x_remain = [i[1:] for i in x]
# 	x_remain_collated = default_collate(x_remain)
# 	return [mpnn_feature] + x_remain_collated






















