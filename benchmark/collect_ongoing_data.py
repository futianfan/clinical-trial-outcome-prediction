'''

input: 	370K data  
	1. ClinicalTrialGov/NCTxxxx/xxxxxx.xml & all_xml    
	1. data/diseases.csv  
	2. data/drug2smiles.pkl          
output: data/raw_data.csv

processing:  
	0.1 Interventional: 273k data (348k total, e.g., observatorial, surgery, )
	0.2 intervention_type == Drug  (drug not empty)
	0.3 drop_set  96k data (273k),  (we don't use drop_set to filter out)
	0.4 -1 -> 0 based on "why_stop" 
	0.5 filter out -1(invalid)

	1. disease -> icd
	2. drug -> smiles  
	3. inclusive / exclusive criteria ---- to do 


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


def get_path_of_all_xml_file():
	input_file = "./data/all_xml"
	with open(input_file, 'r') as fin:
		lines = fin.readlines()
	input_file_lst = [i.strip() for i in lines]
	id_threshold = 4048005 
	input_file_lst = list(filter(lambda x:int(x.split('/')[-1][3:11]) > id_threshold, input_file_lst))
	return input_file_lst 



from utils import walkData

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

def load_disease2icd():
	disease2icd = dict()
	with open('data/diseases.csv', 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	for row in rows:
		disease = row[0]
		icd = row[1]
		disease2icd[disease] = icd 
	return disease2icd 


def nctid2label_dict():
	nctid2outcome = dict() 
	nctid2label = dict() 
	with open("IQVIA/outcome2label.txt", 'r') as fin: 
		lines = fin.readlines() 
		outcome2label = {line.split('\t')[0]:int(line.strip().split('\t')[1]) for line in lines}

	with open("IQVIA/trial_outcomes_v1.csv", 'r') as csvfile: 
		csvreader = list(csv.reader(csvfile))[1:]
		nctid2outcome = {row[0]:row[1] for row in csvreader}

	for nctid,outcome in nctid2outcome.items():
		nctid2label[nctid] = outcome2label[outcome]

	return nctid2label 

nctid2label = nctid2label_dict()



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
		return ("non-Interventional",) 

	interventions = [i for i in root.findall('intervention')]
	drug_interventions = [i.find('intervention_name').text for i in interventions \
														if i.find('intervention_type').text=='Drug']
														# or i.find('intervention_type').text=='Biological']
	if len(drug_interventions)==0:
		return ("Biological",)

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


	##### p-value
	# label = root2outcome(root)  ######## p-value
	# label = -1 if label is None else label 

	##### IQVIA internal data  
	if nctid not in nctid2label:
		label = -1 
	else:
		label = nctid2label[nctid] 

	# if nctid == "NCT00924001":
	# 	print(nctid, label)
	# 	exit()

	try:
		phase = root.find('phase').text 
		# print("phase\n\t\t", phase)
	except:
		phase = ''
	conditions = [i.text for i in root.findall('condition')]

	try:
		criteria = root.find('eligibility').find('criteria').find('textblock').text 
		# print("criteria\n\t\t", criteria)
	except:
		criteria = ''
	#if criteria != '':
	#	assert "Inclusion Criteria:" in criteria 
	#	assert "Exclusion Criteria:" in criteria 
	# title = root.find('brief_title').text	
	# try: 
	# 	summary = root.find('brief_summary').text 
	# 	# print("summary\n\t\t", summary)
	# except:
	# 	summary = '' 

	conditions = [i.lower() for i in conditions]
	drugs = [i.lower() for i in drug_interventions]

	# start_date = root.find("start_date").text 
	# completion_date = root.find("completion_date").text 
	# completion_date_type = root.find("completion_date").type 
	lead_sponsor = root.find('sponsors').find('lead_sponsor').find('agency').text 
	print('lead_sponsor', lead_sponsor)
	try:
		collaborator = root.find('sponsors').find('collaborator').find('agency').text 
		print('collaborator', collaborator)
	except:
		collaborator = '' 

	return nctid, status.lower(), why_stop.lower(), label, phase.lower(), conditions, drugs, criteria, lead_sponsor, collaborator 
	# return nctid, status.lower(), why_stop.lower(), label, phase.lower(), conditions, drugs, title, criteria, summary



def process_all():
	from raw_data_to_feature import load_drug2smiles_pkl, drug_hit_smiles
	### input 
	drug2smiles = load_drug2smiles_pkl()
	disease2icd = load_disease2icd() 
	input_file_lst = get_path_of_all_xml_file()
	### output 
	output_file = 'data/ongoing_data.csv'
	# iqvia_disease2icd, public_disease2icd = load_disease2icd_pkl() 
	# iqvia_disease2diseaseset = disease_dict_reorganize(iqvia_disease2icd)
	# disease2icd = public_disease2icd 
	# disease2diseaseset = disease_dict_reorganize(public_disease2icd)

	t1 = time()
	disease_hit, disease_all, drug_hit, drug_all = 0,0,0,0 ### disease hit icd && drug hit smiles
	# fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 
	# 			 'diseases', 'icdcodes', 'drugs', 'smiless', 
	# 			 'title', 'criteria', 'summary']
	fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 
				 'diseases', 'icdcodes', 'drugs', 'smiless', 
				 'criteria', 'lead_sponsor', 'collaborator']
	num_noninterventional, num_biologics = 0, 0, 
	num_nodrug = 0 
	num_nolabel = 0 
	num_nodisease = 0 
	with open(output_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		data_count = 0
		for file in tqdm(input_file_lst[:]):
			result = xml_file_2_tuple(file)
			## 0.1 & 0.2 
			if len(result)==1 and result[0] == 'non-Interventional':
				num_noninterventional += 1
				continue 
			elif len(result)==1 and result[0]== 'Biological':
				num_biologics += 1
				continue 

			nctid, status, why_stop, label, phase, conditions, drugs, criteria, lead_sponsor, collaborator = result
			# nctid, status, why_stop, label, phase, conditions, drugs, title, criteria, summary = result

			# ## 0.4 
			# if (label == -1) and ('lack of efficacy' in why_stop or 'efficacy concern' in why_stop or \
			# 	'accrual' in why_stop):
			# 	label = 0 
			# ## 0.5
			# if label == -1:
			# 	num_nolabel += 1
			# 	continue 	
			label = 0 


			## 1. disease -> icd
			icdcode_lst = []
			for disease in conditions:
				icdcode = disease2icd[disease] if disease in disease2icd else None
				icdcode_lst.append(icdcode)
			## 2. drug -> smiles 
			smiles_lst = []
			print("drugs ", drugs)
			for drug in drugs:
				# drug_all += 1
				smiles = drug_hit_smiles(drug, drug2smiles)
				if smiles is not None: 
					# drug_hit += 1
					smiles_lst.append(smiles)
				else:
					print("unfounded drug: ", drug)


			if smiles_lst == []:
				num_nodrug += 1
				continue
			icdcode_lst = list(filter(lambda x:x!='None' and x!=None, icdcode_lst))
			if icdcode_lst == []:
				num_nodisease += 1
				continue 

			data_count += 1			
			writer.writerow({'nctid':nctid, \
							 'status': status, \
							 'why_stop': why_stop, \
							 'label':label, \
							 'phase':phase, \
							 'diseases':conditions, \
							 'icdcodes': icdcode_lst, \
							 'drugs':drugs, \
							 'smiless': smiles_lst, \
							 'criteria':criteria, 
							 'lead_sponsor': lead_sponsor, 
							 'collaborator': collaborator})

	t2 = time()
	# print("disease hit icdcode", disease_hit, "disease all", disease_all, "\n drug hit smiles", drug_hit, "drug all", drug_all)
	print(str(int((t2-t1)/60)) + " minutes. " + str(data_count) + " data samples. ")
	print("number of non-Interventional:", num_noninterventional)
	print("number of Biological:", num_biologics)
	print("number of non-label:", num_nolabel)
	print("number of non-drug", num_nodrug)
	print("number of non-disease", num_nodisease)
	return 



## write csv file 
if __name__ == "__main__":
	process_all() 







