# -*- coding: utf-8 -*- 

'''

input: 	348k data  
	1. ClinicalTrialGov/NCTxxxx/xxxxxx.xml 
	2. all_xml

processing:  
	0.1 Interventional: 273k data (348k total, e.g., observatorial, surgery, )
	0.2 intervention_type == Drug  (drug not empty)
	0.3 drop_set  96k data (273k),  (we don't use drop_set to filter out)
	0.4 -1 -> 0 based on "why_stop" 
	0.5 filter out -1(invalid)

	xxxxxxx


	1. disease -> icd

output:
	1. 	output_file = 'data/diseases.csv' 

requires ~10 minutes. 

'''

##### standard library
import os, csv, pickle   
from xml.dom import minidom
from xml.etree import ElementTree as ET
from collections import defaultdict
from time import time 
import re 
from tqdm import tqdm 
import requests

from utils import get_path_of_all_xml_file, walkData

drop_set = ['Active, not recruiting', 'Enrolling by invitation', 'No longer available',  
			'Not yet recruiting', 'Recruiting', 'Temporarily not available', 'Unknown status']

'''
14 overall_status 
	
	 Active, not recruiting
	 Approved for marketing
	 Available
	 Completed
	 Enrolling by invitation
	 No longer available
	 Not yet recruiting
	 Recruiting
	 Suspended
	 Temporarily not available
	 Terminated
	 Unknown status
	 Withdrawn
	 Withheld
'''




### tricky 
def normalize_disease(name):
	name = name.lower()
	if 'lymphoma' in name:
		return 'lymphoma'
	name = name.replace(',', '')
	name = name.replace('(', ' ')
	name = name.replace(')', ' ')

	name = name.replace('cancer', 'neoplasm')
	name = name.replace('neoplasms', 'neoplasm')
	name = name.replace('tumors', 'tumor')

	name = name.replace('infections', 'infection')
	name = name.replace('diseases', 'disease')
	name = name.replace('disorders', 'disorder')
	name = name.replace('syndromes', 'syndrome')

	name = ' '.join(name.split())
	if name.split()[0]=='stage':
		name = ' '.join(name.split()[2:])

	name_lst = [name]
	if ' neoplasm' in name:
		print(name)
		name_lst.append(name.replace('neoplasm', 'tumor'))
		name_split = name.split()
		idx = name_split.index('neoplasm')
		name2 = name_split[idx-1] + ' ' + name_split[idx]
		name_lst.append(name2)
	if ' tumor' in name:
		name_lst.append(name.replace('tumor', 'neoplasm'))
		name_split = name.split()
		idx = name_split.index('tumor')
		name2 = name_split[idx-1] + ' ' + name_split[idx]
		name_lst.append(name2)	
	if 'disease' in name:
		name_lst.append(name.replace('disease', '').strip())
	if 'disorder' in name:
		name_lst.append(name.replace('disorder', '').strip())
	if '-related' in name:
		name_lst.append(name.replace('-related', '').strip())
	if 'syndrome' in name:
		name_lst.append(name.replace('syndrome', '').strip())


	if 'lung' in name and 'carcinoma' in name:
		name_lst.append('lung carcinoma')
	elif 'cell' in name and 'carcinoma' in name:
		name_lst.append('cell carcinoma')
	elif 'carcinoma' in name:
		name_lst.append('carcinoma')



	## approximation 1	very few 
	organ = ['liver', 'kidney', 'cardio', 'renal', 'hiv']
	for word in organ:
		if word in name:
			name_lst.append(word)

	# approximation 2 most 20% 
	word_lst = sorted([(word, len(word)) for word in name.split()], key = lambda x:x[1], reverse = True)
	for word, cnt in word_lst:
		if cnt < 8:
			break 
		name_lst.append(word)

	return name_lst



def get_icd_from_nih(name):
	prefix = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms='
	name_lst = normalize_disease(name)
	for name in name_lst:
		url = prefix + name 
		response = requests.get(url)
		text = response.text 
		if text == '[0,[],null,[]]':
			continue  
		text = text[1:-1]
		idx1 = text.find('[')
		idx2 = text.find(']')
		codes = text[idx1+1:idx2].split(',')
		codes = [i[1:-1] for i in codes]
		return codes 
	return None 

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


def xml_file_2_tuple(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	nctid = root.find('id_info').find('nct_id').text	### nctid: 'NCT00000102'
	study_type = root.find('study_type').text 
	if study_type != 'Interventional':
		return (None,)  ### invalid 

	interventions = [i for i in root.findall('intervention')]
	drug_interventions = [i.find('intervention_name').text for i in interventions \
														if i.find('intervention_type').text=='Drug']
														# or i.find('intervention_type').text=='Biological']
	if len(drug_interventions)==0:
		return (None,)

	try:
		status = root.find('overall_status').text 
	except:
		status = ''
	# if status in drop_set:
	# 	return (None,)  ### invalid 
	try:
		why_stop = root.find('why_stopped').text
	except:
		why_stop = ''
	label = root2outcome(root)
	label = -1 if label is None else label 


	conditions = [i.text for i in root.findall('condition')]
	conditions = [i.lower() for i in conditions]
	return conditions, label, why_stop, None



















def process_all():
	output_file = 'data/diseases.csv'
	t1 = time()
	disease_hit, disease_all = 0,0 ### disease hit icd && drug hit smiles
	input_file_lst = get_path_of_all_xml_file()
	disease2icd_and_cnt = dict()
	unfounded_disease_cnt = defaultdict(int)
	word_cnt = defaultdict(int)
	fieldname = ['disease', 'icd', 'count']

	data_count = 0
	for file in tqdm(input_file_lst[:]):
		result = xml_file_2_tuple(file)
		## 0.1 & 0.2 
		if len(result)==1:
			continue 	### only interventions
		conditions, label, why_stop, _ = result
		## 0.4 
		if (label == -1) and ('lack of efficacy' in why_stop or 'efficacy concern' in why_stop or \
			'accrual' in why_stop):
			label = 0 
		## 0.5
		if label == -1:
			continue 

		data_count += 1	
		icdcode_lst = []
		for disease in conditions:
			disease_all += 1
			disease_hit += 1
			if disease in disease2icd_and_cnt:
				disease2icd_and_cnt[disease][1] += 1
				if disease2icd_and_cnt[disease][0] == 'None':
					disease_hit -= 1 
					unfounded_disease_cnt[disease] += 1 
			else:				
				codes = get_icd_from_nih(disease)
				if codes is None:
					disease2icd_and_cnt[disease] = ['None', 1]
					disease_hit -= 1
					unfounded_disease_cnt[disease] += 1
				else:
					disease2icd_and_cnt[disease] = [codes, 1] 
	t2 = time()
	disease2cnt = sorted([(k,v) for k,v in unfounded_disease_cnt.items()], key = lambda x:x[1], reverse = True)
	for disease, cnt in disease2cnt:
		for word in disease.split():
			word_cnt[word] += cnt

	disease_icd_cnt = sorted([[disease,icd,cnt] for disease,(icd,cnt) in disease2icd_and_cnt.items()], key = lambda x:x[2], reverse=True)



	### output 
	with open(output_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		for disease, icd, cnt in disease_icd_cnt:
			writer.writerow({'disease':disease, 'icd':icd, 'count':cnt})


	### use for debug 
	with open('unfounded_disease_cnt.txt', 'w') as fout:
		for disease, cnt in disease2cnt:
			fout.write(disease + '\t\t' + str(cnt) + '\n')
		fout.write('\n'*10)
		word_cnt = sorted([(w,c) for w,c in word_cnt.items()], key = lambda x:x[1], reverse = True)
		for word, cnt in word_cnt:
			fout.write(word + '\t\t' + str(cnt) + '\n')

	print("disease hit icdcode", disease_hit, "disease all", disease_all)
	print(str(int((t2-t1)/60)) + " minutes. " + str(data_count) + " data samples. ")
	return 




## write csv file 
if __name__ == "__main__":
	process_all() 







