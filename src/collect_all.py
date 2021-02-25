# -*- coding: utf-8 -*- 
import os, csv, pickle   
from xml.dom import minidom
from xml.etree import ElementTree as ET
from collections import defaultdict
from time import time 
import re 
from tqdm import tqdm 

from utils import dynamic_programming


def get_all_file():
	input_file = "all_xml"
	with open(input_file, 'r') as fin:
		lines = fin.readlines()
	input_file_lst = [i.strip() for i in lines]
	return input_file_lst 

'''
input_file_lst = [ 
	'ClinicalTrialGov/NCT0000xxxx/NCT00000102.xml', 
 	'ClinicalTrialGov/NCT0000xxxx/NCT00000104.xml', 
 	'ClinicalTrialGov/NCT0000xxxx/NCT00000105.xml', 
	  ... ]
'''

def remove_multiple_space(text):
	text = ' '.join(text.split())
	return text 

def generate_complete_path(nctid):
	assert len(nctid)==11
	prefix = nctid[:7] + "xxxx"
	datafolder = os.path.join("./ClinicalTrialGov/", prefix, nctid+".xml")
	return datafolder 

#  xml read blog:  https://blog.csdn.net/yiluochenwu/article/details/23515923 
def walkData(root_node, prefix, result_list):
	temp_list =[prefix + '/' + root_node.tag, root_node.text]
	result_list.append(temp_list)
	children_node = root_node.getchildren()
	if len(children_node) == 0:
		return
	for child in children_node:
		walkData(child, prefix = prefix + '/' + root_node.tag, result_list = result_list)

def root2outcome(root):
	result_list = []
	walkData(root, prefix = '', result_list = result_list) 
	filter_func = lambda x:'p_value' in x[0] 
	outcome_list = list(filter(filter_func, result_list))
	if len(outcome_list)==0:
		return None 
	outcome = outcome_list[0][1]
	if outcome[0]=='<':
		return 1
	if outcome[0]=='>':
		return 0 
	if outcome[0]=='=':
		outcome = outcome[1:]
	try:
		label = float(outcome)
		if label < 0.05:
			return 1
		else:
			return 0
	except:
		return None 

def file2dict(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	nctid = root.find('id_info').find('nct_id').text	### nctid: 'NCT00000102'
	title = root.find('brief_title').text
	study_type = root.find('study_type').text 
	if study_type != 'Interventional':
		return (None,)
	label = root2outcome(root)
	if label is None:
		return (None,)
	conditions = [i.text for i in root.findall('condition')]
	interventions = [i for i in root.findall('intervention')]
	drug_interventions = [i.find('intervention_name').text for i in interventions \
														if i.find('intervention_type').text=='Drug']
														# or i.find('intervention_type').text=='Biological']
	#print(len(interventions), "drug intervention", drug_interventions)
	try:
		status = root.find('overall_status').text 
	except:
		status = ''
	try:
		criteria = root.find('eligibility').find('criteria').find('textblock').text 
		print("criteria\n\t\t", criteria)
	except:
		criteria = ''
	#if criteria != '':
	#	assert "Inclusion Criteria:" in criteria 
	#	assert "Exclusion Criteria:" in criteria 
	try: 
		summary = root.find('brief_summary').text 
		print("summary\n\t\t", summary)
	except:
		summary = '' 
	try:
		phase = root.find('phase').text 
		print("phase\n\t\t", phase)
	except:
		phase = ''
	return nctid, status, label, phase, conditions, drug_interventions, title, criteria, summary 



def getXmlData(file_name):
	result_list = []
	root = ET.parse(file_name).getroot()
	walkData(root, prefix = '', result_list = result_list) 
	return result_list


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
	public_pkl_file = "icdcode/description2icd.pkl"
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
			#print("drug hit: ", drug, '&', word)
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
			# print("I disease hit:", disease, '&', word)
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
		#print("II disease hit:", disease, '&', best_disease)		
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



def write_csv_file():
	cook_csv_file = 'data/cooked_trial.csv'
	drug2smiles = load_drug2smiles_pkl()
	iqvia_disease2icd, public_disease2icd  = load_disease2icd_pkl() 
	iqvia_disease2diseaseset = disease_dict_reorganize(iqvia_disease2icd)
	disease2icd = public_disease2icd 
	disease2diseaseset = disease_dict_reorganize(public_disease2icd)
	t1 = time()
	input_file_lst = get_all_file()
	fieldname = ['nctid', 'status', 'label', 'phase', 'diseases', 'icdcodes', 'drugs', 'smiless', 'title', 'criteria', 'summary']
	disease_hit, disease_all, drug_hit, drug_all = 0,0,0,0 ### disease hit icd && drug hit smiles
	with open(cook_csv_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		for file in tqdm(input_file_lst[:]):
			result = file2dict(file)
			if len(result)==1:
				continue 
			nctid, status, label, phase, diseases, drugs, title, criteria, summary = result
			icdcode_lst, smiles_lst = [], []
			for disease in diseases:
				disease = disease.lower()
				disease_all += 1
				icdcode = disease_hit_icd(disease, disease2icd, disease2diseaseset)
				if icdcode is not None:
					disease_hit += 1
					icdcode_lst.append(icdcode)
				else:
					print("unfounded:", disease)
			if len(icdcode_lst)==0:
				continue  
			for drug in drugs:
				drug = drug.lower()
				drug_all += 1
				smiles = drug_hit_smiles(drug, drug2smiles)
				if smiles is not None: 
					drug_hit += 1
					smiles_lst.append(smiles)
			if len(smiles_lst)==0:
				continue
			icdcodes = '\t'.join(icdcode_lst)
			smiless = '\t'.join(smiles_lst)
			drugs = '\t'.join(smiles_lst)
			diseases = '\t'.join(diseases)
			writer.writerow({'nctid':nctid, \
							 'label':label, \
							 'phase':phase, \
							 'diseases':diseases.encode('utf-8'), \
							 'icdcodes': icdcodes, \
							 'drugs':drugs.encode('utf-8'), \
							 'smiless': smiless, \
							 'title':title.encode('utf-8'), \
							 'criteria':criteria.encode('utf-8'), \
							 'summary':summary.encode('utf-8')})
	print("disease hit icdcode", disease_hit, "disease all", disease_all, "\n drug hit smiles", drug_hit, "drug all", drug_all)
	t2 = time()
	print(str(int((t2-t1)/60)) + " minutes")
	return 


## dynamic programming
# if __name__ == "__main__":
# 	a = 'dynamdddwic'
# 	b = 'mfewweic'
# 	print(dynamic_programming(a,b))

## write csv file 
if __name__ == "__main__":
	write_csv_file() 

# #### check csvfile
# if __name__ == "__main__":
# 	cook_csv_file = 'data/cooked_trial.csv'
# 	positive_sample_cnt, negative_sample_cnt = 0, 0
# 	wrong_nct_list = []
# 	correct_cnt, total_cnt = 0, 0 
# 	iqvia_nct2label = Get_Iqvia_data() 
# 	with open(cook_csv_file, 'r') as csvfile:
# 		reader = list(csv.reader(csvfile, delimiter = ','))[1:]
# 		for row in reader:
# 			nctid = row[0]
# 			label = int(row[1])
# 			if nctid in iqvia_nct2label:
# 				total_cnt += 1
# 				iqvia_label = iqvia_nct2label[nctid]
# 				if iqvia_label == label:
# 					correct_cnt += 1
# 				else:
# 					wrong_nct_list.append(nctid)
# 			if label == 1:
# 				positive_sample_cnt += 1
# 			elif label==0:
# 				negative_sample_cnt += 1 
# 	print("positive_sample_cnt", positive_sample_cnt, "negative_sample_cnt", negative_sample_cnt)
# 	print("correct_cnt", correct_cnt, "total_cnt", total_cnt)
# 	with open("wrong_nct.txt", 'w') as fout:
# 		for nctid in wrong_nct_list:
# 			fout.write(nctid + '\n')



##### p_value 
# if __name__ == "__main__":
# 	##### server
# 	nctid = "NCT00001723"
# 	file = generate_complete_path(nctid)
# 	### local 
# 	file = "NCT00001723.xml" 

# 	input_file_lst = get_all_file() 
# 	for file in input_file_lst[:100000]:
# 		result_list = getXmlData(file)
# 		filter_func = lambda x:'p_value' in x[0] 
# 		outcome_list = list(filter(filter_func, result_list))
# 		if len(outcome_list) > 0:
# 			print('='*50)
# 			print(file.split('/')[-1].split('.')[0])
# 			for i in outcome_list:
# 				print(i)




