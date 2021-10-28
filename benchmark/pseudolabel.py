# -*- coding: utf-8 -*- 

'''

input: 	348k data  
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

def xmlfile_2_label(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	nctid = root.find('id_info').find('nct_id').text	### nctid: 'NCT00000102'

	label = root2outcome(root)
	label = -1 if label is None else label 

	return label 






