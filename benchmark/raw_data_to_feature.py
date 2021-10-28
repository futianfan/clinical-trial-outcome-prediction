'''
input:
	1. data/raw_data.csv 

process:
	0. filter out invalid
	1. disease -> icd
	2. drug -> smiles  
	3. inclusive / exclusive criteria 



output:
	1. data/feature.csv 	


'''


# -*- coding: utf-8 -*- 
import os, csv, pickle, re    
from xml.dom import minidom
from xml.etree import ElementTree as ET
from collections import defaultdict
from time import time 
from tqdm import tqdm 



def Get_Iqvia_data():
	nct2outcome_file = "data/trial_outcomes_v1.csv"
	outcome2label_file = "data/outcome2label.txt"
	outcome2label = dict()
	nct2label = dict() 
	with open(outcome2label_file, 'r') as fin:
		lines = fin.readlines() 
	for line in lines:
		outcome = line.split('\t')[0]
		label = int(line.split('\t')[1])
		outcome2label[outcome] = label 
	with open(nct2outcome_file, 'r') as csvfile:
		reader = list(csv.reader(csvfile, delimiter=','))[1:]
		for row in reader:
			nctid, outcome = row[0], row[1]
			label = outcome2label[outcome]
			if nctid in nct2label:
				if label > nct2label[nctid]:
					nct2label[nctid] = label 
			else:
				nct2label[nctid] = label 
	### remove the key whole value is -1
	for nctid in list(nct2label.keys()):
		label = nct2label[nctid]
		if label == -1:
			nct2label.pop(nctid)
	return nct2label 

def load_drug2smiles_pkl():
	pkl_file = "data/drug2smiles.pkl"
	drug2smiles = pickle.load(open(pkl_file, 'rb'))
	return drug2smiles 

def load_disease2icd_pkl():
	iqvia_pkl_file = "data/disease2icd.pkl"
	public_pkl_file = "icdcode/description2icd10.pkl"
	iqvia_disease2icd = pickle.load(open(iqvia_pkl_file, 'rb'))
	public_disease2icd = pickle.load(open(public_pkl_file, 'rb'))
	return iqvia_disease2icd, public_disease2icd 



def drug_hit_smiles(drug, drug2smiles):
	"""
		heuristics
	"""
	if drug in drug2smiles:
		return drug2smiles[drug]
	for word in drug.split():
		if len(word)>=7 and word in drug2smiles:
			print("drug hit: ", drug, '&', word)
			return drug2smiles[word]
	# max_length = 0
	# for drug0 in drug2smiles:
	# 	length = dynamic_programming(drug, drug0)
	# 	if length > max_length:
	# 		best_drug = drug0 
	# 		max_length = length 
	# if max_length > 9: 
	# 	print("DP drug hit: ", drug, '&', best_drug)
	# 	return drug2smiles[best_drug]
	return None 		


def disease_hit_icd(disease, disease2icd, disease2diseaseset):
	"""
		heuristics
	"""
	#### match 0
	if disease in disease2icd:
		return disease2icd[disease]
	#### match 1
	for word in disease.split():
		if len(word)>=7 and word in disease2icd:
			print("I disease hit:", disease, '&', word)
			return disease2icd[word]
	#### match 2
	max_length = 0
	diseaseset = set(re.split(r"[\', /-]",disease))
	for disease0, disease0set in disease2diseaseset.items():
		intersection_set = disease0set.intersection(diseaseset)
		length = len(intersection_set)
		wordlength = len(''.join(list(intersection_set)))
		if length > max_length and wordlength > 8:
			max_length = length
			best_disease = disease0
	if max_length > 1:
		print("II disease hit:", disease, '&', best_disease)		
		return disease2icd[best_disease]

	# max_length = 0
	# for disease0 in disease2icd:
	# 	length = dynamic_programming(disease, disease0)
	# 	if length > max_length:
	# 		best_disease = disease0 
	# 		max_length = length 
	# if max_length > 20: 
	# 	print("III DP disease hit: ", disease, '&', best_disease)
	# 	return disease2icd[best_disease]	
	return None


def disease_dict_reorganize(disease2icd):
	return {disease:set(re.split(r"[\', /-]",disease)) for disease in disease2icd}




# def main(reader, feature_file):
# 	drug2smiles = load_drug2smiles_pkl()
# 	iqvia_disease2icd, public_disease2icd = load_disease2icd_pkl() 
# 	iqvia_disease2diseaseset = disease_dict_reorganize(iqvia_disease2icd)
# 	disease2icd = public_disease2icd 
# 	disease2diseaseset = disease_dict_reorganize(public_disease2icd)
# 	t1 = time()
# 	fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 
# 				 'diseases', 'icdcodes', 'drugs', 'smiless', 'title', 'criteria', 'summary']
# 	disease_hit, disease_all, drug_hit, drug_all = 0,0,0,0 ### disease hit icd && drug hit smiles
# 	with open(feature_file, 'w') as csvfile:
# 		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
# 		writer.writeheader()
# 		for row in reader: 
# 			nctid, status, why_stop, label, phase, conditions, drugs, title, criteria, summary = row 
# 			print(nctid)
# 			## 0. filter out invalid
# 			if (label == -1) and ('lack of efficacy' in why_stop.lower() or 'efficacy concern' in why_stop.lower() or \
# 				'accrual' in why_stop.lower()):
# 				label = 0 
# 			if label == -1:
# 				continue 	

# 			## 1. disease -> icd
# 			icdcode_lst = []
# 			for disease in conditions.split('\t'):
# 				disease = disease.lower()
# 				disease_all += 1
# 				icdcode = disease_hit_icd(disease, disease2icd, disease2diseaseset)
# 				if icdcode is not None:
# 					disease_hit += 1
# 					icdcode_lst.append(icdcode)
# 				else:
# 					print("unfounded disease:  ", disease) 

# 			## 2. drug -> smiles 

# 			smiles_lst = []
# 			print("drugs ", drugs)
# 			for drug in drugs.split('\t'):
# 				drug = drug.lower()
# 				drug_all += 1
# 				smiles = drug_hit_smiles(drug, drug2smiles)
# 				if smiles is not None: 
# 					drug_hit += 1
# 					smiles_lst.append(smiles)
# 				else:
# 					print("unfounded drug:  ", drug)

# 			## 3. inclusion / exclusion criteria 
# 			pass 


# 			icdcodes = '\t'.join(icdcode_lst)
# 			smiless = '\t'.join(smiles_lst)
# 			writer.writerow({'nctid':nctid, \
# 							 'status': status, \
# 							 'why_stop': why_stop, \
# 							 'label':label, \
# 							 'phase':phase, \
# 							 'diseases':conditions, \
# 							 'icdcodes': icdcodes, \
# 							 'drugs':drugs, \
# 							 'smiless': smiless, \
# 							 'title':title, \
# 							 'criteria':criteria, \
# 							 'summary':summary})
# 	print("disease hit icdcode", disease_hit, "disease all", disease_all, "\n drug hit smiles", drug_hit, "drug all", drug_all)
# 	t2 = time()
# 	print(str(int((t2-t1)/60)) + " minutes")
# 	return 


# if __name__ == "__main__":
# 	raw_data_file = "data/raw_data.csv" 
# 	feature_file = "data/feature.csv"
# 	with open(raw_data_file, 'r') as csvfile:
# 		reader = list(csv.reader(csvfile, delimiter = ','))[1:]
# 	main(reader, feature_file) 


















