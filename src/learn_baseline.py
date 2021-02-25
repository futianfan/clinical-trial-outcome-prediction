## 1. import 
## 2. input & hyperparameter
## 3. pretrain 
## 4. 'dataloader, model build, train, inference'
################################################


## 1. import 
import torch, os 
import numpy as np 
from functools import reduce 
from dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst, csv_three_feature_2_complete_dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from rdkit.Chem import AllChem
from rdkit import Chem 


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp

def smiles2fp(smiles):
	try:
		mol = Chem.MolFromSmiles(smile)
		fp = fingerprints_from_mol(mol)
		return fp 
	except:
		return np.zeros((1, 2048), np.int32)

def smiles_lst2fp(smiles_lst):
	fp_lst = [smiles2fp(smiles) for smiles in smiles_lst]
	fp_mat = np.concatenate(fp_lst, 0)
	fp = np.mean(fp_mat,0)
	return fp	


## 2. input & hyperparameter
base_name_lst = ['trial', 'phase_I', 'phase_II', 'phase_III']

base_name = 'trial' 
assert base_name in base_name_lst 
data_folder = './ctgov_data'
train_file = os.path.join(data_folder, base_name + '_train.csv')
valid_file = os.path.join(data_folder, base_name + '_valid.csv')
test_file = os.path.join(data_folder, base_name + '_test.csv')





## dataloader -> feature vector 
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 
# test_complete_loader = csv_three_feature_2_complete_dataloader(test_file, shuffle=False, batch_size = 32)


icdcode_lst3s = [icdcode_lst3 for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in train_loader]
icdcode_lst3 = list(reduce(lambda x,y:x+y, icdcode_lst3s))

global_icd = set()
for lst2 in icdcode_lst3:
	lst = list(reduce(lambda x,y:x+y, lst2))
	lst = [i.split('.')[0] for i in lst]
	global_icd = global_icd.union(set(lst))	
global_icd = list(global_icd)
num_icd = len(global_icd)




def dataloader2Xy(dataloader):
	labels = [label_vec for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in dataloader]
	labels = torch.cat(labels)  ### shape: (n,)  
	y = np.array(labels)


	smiless = [smiles_lst2 for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in dataloader]
	smiles_lst2 = list(reduce(lambda x,y:x+y, smiless))
	fp_lst = [smiles_lst2fp(smiles_lst).reshape(1,-1) for smiles_lst in smiles_lst2]
	fp_mat = np.concatenate(fp_lst, 0)


	icdcode_lst3s = [icdcode_lst3 for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in dataloader]
	icdcode_lst3 = list(reduce(lambda x,y:x+y, icdcode_lst3s))
	icdcode_lst = []
	for lst2 in icdcode_lst3:
		lst = list(reduce(lambda x,y:x+y, lst2))
		lst = [i.split('.')[0] for i in lst]
		lst = set(lst)	
		icd_feature = np.zeros((1,num_icd), np.int32)
		for ele in lst:
			if ele in global_icd:
				idx = global_icd.index(ele)
				icd_feature[0,idx] = 1 
		icdcode_lst.append(icd_feature)
	icdcode_mat = np.concatenate(icdcode_lst, 0)


	criteria_lsts = [criteria_lst for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in dataloader]
	criteria_lst = list(reduce(lambda x,y:x+y, criteria_lsts))

	X = np.concatenate([fp_mat, icdcode_mat], 1)
	return X, y 


train_X, train_y = dataloader2Xy(train_loader)
test_X, test_y = dataloader2Xy(test_loader)




# model build, train, inference
model_lst = [LogisticRegression, RandomForestClassifier, AdaBoostClassifier]

clf = LogisticRegression(random_state=0).fit(train_X, train_y)
prediction_float = clf.predict_proba(test_X)
prediction_float = prediction_float[:,1]
prediction_binary = clf.predict(test_X)
auc_score = roc_auc_score(test_y, prediction_float)
f1score = f1_score(test_y, prediction_binary)
prauc_score = average_precision_score(test_y, prediction_binary)
print(auc_score, f1score, prauc_score)


clf = RandomForestClassifier(random_state=0).fit(train_X, train_y)
prediction_float = clf.predict_proba(test_X)
prediction_float = prediction_float[:,1]
prediction_binary = clf.predict(test_X)
auc_score = roc_auc_score(test_y, prediction_float)
f1score = f1_score(test_y, prediction_binary)
prauc_score = average_precision_score(test_y, prediction_binary)
print(auc_score, f1score, prauc_score)


clf = AdaBoostClassifier(random_state=0).fit(train_X, train_y)
prediction_float = clf.predict_proba(test_X)
prediction_float = prediction_float[:,1]
prediction_binary = clf.predict(test_X)
auc_score = roc_auc_score(test_y, prediction_float)
f1score = f1_score(test_y, prediction_binary)
prauc_score = average_precision_score(test_y, prediction_binary)
print(auc_score, f1score, prauc_score)










