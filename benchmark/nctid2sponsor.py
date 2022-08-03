'''
sponsor statistics for all interventional clinical trials 


data/nctid2sponsor.csv
	- nctid,sponsor

data/sponsor2count.csv 
	- sponsor,count 

data/sponsor2approvalrate.csv 
	- sponsor,approval_rate

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
	return input_file_lst 

from utils import walkData
drop_set = ['Active, not recruiting', 'Enrolling by invitation', 'No longer available',  
			'Not yet recruiting', 'Recruiting', 'Temporarily not available', 'Unknown status'] 






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

	return nctid, lead_sponsor, collaborator 


sponsor2cnt = defaultdict(lambda: 0)

def process_all():
	input_file_lst = get_path_of_all_xml_file()
	### output 
	output_file = 'data/nctid2sponsor.csv'
	sponsor2cnt_file = 'data/sponsor2count.csv'
	disease_hit, disease_all, drug_hit, drug_all = 0,0,0,0 ### disease hit icd && drug hit smiles
	# fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 
	# 			 'diseases', 'icdcodes', 'drugs', 'smiless', 
	# 			 'title', 'criteria', 'summary']
	fieldname = ['nctid', 'lead_sponsor']
	with open(output_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		for file in tqdm(input_file_lst[:]):
			result = xml_file_2_tuple(file)
			if len(result) == 1:
				continue 
			nctid, lead_sponsor, collaborator = result

			writer.writerow({'nctid':nctid, 
							 'lead_sponsor': lead_sponsor,})
			sponsor2cnt[lead_sponsor] += 1 
	sponsor_count_list = [(sponsor,count) for sponsor, count in sponsor2cnt.items()]
	sponsor_count_list.sort(key=lambda x:x[1], reverse = True)
	fieldname = ['sponsor', 'count']
	with open(sponsor2cnt_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		for sponsor,count in sponsor_count_list:
			writer.writerow({'sponsor':sponsor, 'count':str(count)})
	return 

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




def sponsor2approvalrate():
	nctid2label = nctid2label_dict()
	sponsor2success_total = defaultdict(lambda: [0,0])
	for nctid, label in tqdm(nctid2label.items()):
		xml_file = 'ctgov/' + nctid[:7] + 'xxxx/' + nctid + '.xml'
		if not os.path.exists(xml_file): 
			continue 
		tree = ET.parse(xml_file)
		root = tree.getroot()
		lead_sponsor = root.find('sponsors').find('lead_sponsor').find('agency').text 
		sponsor2success_total[lead_sponsor][1] += 1
		if label == 1:
			sponsor2success_total[lead_sponsor][0] += 1 
	sponsor_approvalrate_total = [[sponsor, success / total, total] for sponsor, (success, total) in sponsor2success_total.items()]	
	sponsor_approvalrate_total.sort(key = lambda x:x[2], reverse=True)

	with open('data/sponsor2approvalrate.csv', 'w') as csvfile:
		fieldname = ['sponsor', 'approval_rate', 'total']
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()
		for sponsor,approval_rate, total in sponsor_approvalrate_total:
			writer.writerow({'sponsor':sponsor, 'approval_rate':str(approval_rate), 'total': str(total)})

	sponsor_approvalrate_total = list(filter(lambda x:x[2]>50,sponsor_approvalrate_total))
	sponsor_approvalrate_total.sort(key = lambda x:x[1], reverse = True)
	print(sponsor_approvalrate_total[:10])

# process_all() 
sponsor2approvalrate() 






