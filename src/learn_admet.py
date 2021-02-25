## 1. import 
## 2. input & hyperparameter
## 3. utility 
## 4. <5 line for 'dataloader, model build, train, inference'
################################################
## 1. import 
#### standard module
import os, torch, numpy as np, pickle, sys
from torch.utils.data import Dataset, DataLoader
sys.setrecursionlimit(10000)
torch.manual_seed(0)


#### local module
from models import SMILES_Classifier
from data_loader import SMILESData
from utils import smiles2morgan 
################################################
## 2. input & hyperparameter
num_workers = 3
batch_size = 32
mole_vec_dim = 1024 
train_data_params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}
hidden_size_lst = [200, 1]
data_folder = "ADMET/cooked/"
admet_lst = ["absorption", "distribution", "metabolism", "excretion", "toxicity"]
################################################################################
## 3. utility 
def SMILES2vec(smiles):
	morgan_feature = smiles2morgan(smiles)
	morgan_feature = torch.from_numpy(morgan_feature).float() 
	return morgan_feature 

def file2dataset(file):
	with open(file, 'r') as fin:
		lines = fin.readlines()
	lines = [line.strip().split("\t") for line in lines]
	for i in lines:
		if len(i)==1:
			print(i)
	smiles_lst = [i[0] for i in lines]
	label_lst = [int(i[1]) for i in lines]
	dataset = SMILESData(smiles_lst = smiles_lst, label_lst = label_lst, smiles2vec=SMILES2vec)
	dataloader = DataLoader(dataset, **train_data_params)
	return dataloader 

def file2dataloader(file_prefix):
	'''
	input: 
		file's name
	output:
		train's dataloader 
		valid's dataloader
	'''
	train_file = file_prefix + "_train.txt"
	valid_file = file_prefix + "_valid.txt"
	train_data_loader = file2dataset(train_file)
	valid_data_loader = file2dataset(valid_file)
	return train_data_loader, valid_data_loader 
################################################################################ 
## 4. <5 line for 'dataloader, model build, train, inference'
result_lst = []
for task in admet_lst:
	prefix = os.path.join(data_folder, task)
	print("="*20 + task + "="*20)
	train_data_loader, valid_data_loader = file2dataloader(prefix)
	model = SMILES_Classifier(mole_vec_dim = mole_vec_dim, hidden_size_lst = hidden_size_lst)
	print(task, "hidden size", model.hidden_size)
	#auc_score, f1score, prauc_score = model.train(train_loader = train_data_loader, valid_loader = valid_data_loader, saved_name = "figure/" + task)
	#result_lst.append((auc_score, f1score, prauc_score))
	save_path = os.path.join("save_model", task + ".ckpt")
	torch.save(dict(model = model, model_state = model.state_dict()), save_path)

exit()
for task, (auc_score, f1score, prauc_score) in zip(admet_lst, result_lst):
	print("="*20 + task + "="*20)
	print("RUC-AUC:", str(auc_score)[:6], "f1-Score:", str(f1score)[:6], "PR-AUC:", str(prauc_score)[:6])









