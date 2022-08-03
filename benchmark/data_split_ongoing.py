# -*- coding: utf-8 -*- 

'''

input: 	9k data ?  
	1. ctgov_data/raw_data.csv 

nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria

processing:  
	1. phase I
	2. phase II
	3. phase III
	4. indication 
	5. train/test split 


output:
	1. ctgov_data/phase_I.csv 
	2. ctgov_data/phase_II.csv 
	3. ctgov_data/phase_III.csv 
	4. ctgov_data/trial.csv 

requires ~10 minutes. 

'''

import csv 
from random import shuffle 
## no shuffle 
from functools import reduce


from ccs_utils import file2_icd2ccs_and_ccs2description, file2_icd2ccsr
# icd2ccs, ccscode2description = file2_icd2ccs_and_ccs2description() 
icd2ccsr = file2_icd2ccsr()


def csvfile2rows(input_file):
	with open(input_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	return rows 

def filter_phase_I(row):
	if "phase 1" in row[4]:
		return True	
	## enhance 
	# if int(row[3])==1 and row[4]=='phase 4':
	# 	return True 

	return False 

def filter_phase_II(row):
	phase = row[4] 
	label = int(row[3])

	if "phase 2" in row[4]:
		return True
	## enhance
	# if int(row[3])==1 and 'phase 4' in row[4]:
	# 	return True

	return False

def filter_phase_III(row):
	if "phase 3" in row[4]:
		return True
	### enhance 
	# if "phase 4" in row[4] and int(row[3])==1:
	# 	return True
	# if int(row[3])==0 and row[4] =='phase 2':
	# 	return True 
	return False 

def filter_trial(row):
	label = int(row[3])
	if label == 0 and ('phase 1' in row[4] or 'phase 2' in row[4]):
		return True 
	if ('phase 3' in row[4] or 'phase 4' in row[4]) and label==1:  ### label == 1
		return True 
	return False 

# def filter_chronic(row):
# 	if 'chronic' in row[5]:
# 		return True 
# 	return False 

# def filter_cardio(row):
# 	if 'cardio' in row[5]:
# 		return True 
# 	return False 

# def filter_cancer(row):
# 	if 'cancer' in row[5] or 'neoplasm' in row[5] or 'tumor' in row[5]:
# 		return True 
# 	return False 

# def filter_pain(row):
# 	if 'pain' in row[5]:
# 		return True 
# 	return False 

def icdcode_text_2_lst_of_lst(text):
	text = text[2:-2]
	lst_lst = []
	for i in text.split('", "'):
		i = i[1:-1]
		lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
	return lst_lst 

def row2icdcodelst(row):
	icdcode_text = row[6]
	icdcode_lst2 = icdcode_text_2_lst_of_lst(icdcode_text)
	icdcode_lst = reduce(lambda x,y:x+y, icdcode_lst2)
	icdcode_lst = [i.replace('.', '') for i in icdcode_lst]
	return icdcode_lst 




# def filter_heart(row):
# 	icdcode_text = row[6]
# 	icdcode_lst2 = icdcode_text_2_lst_of_lst(icdcode_text)
# 	icdcode_lst = reduce(lambda x,y:x+y, icdcode_lst2)
# 	icdcode_lst = [i.replace('.', '') for i in icdcode_lst]
# 	for icdcode in icdcode_lst:
# 		try:
# 			ccs = icd2ccs[icdcode]
# 			description = ccscode2description[ccs].lower() 	
# 			if 'heart' in description:
# 				return True 
# 		except:
# 			pass 
# 	return False 

# def filter_infection(row):
# 	icdcode_text = row[6]
# 	icdcode_lst2 = icdcode_text_2_lst_of_lst(icdcode_text)
# 	icdcode_lst = reduce(lambda x,y:x+y, icdcode_lst2)
# 	icdcode_lst = [i.replace('.', '') for i in icdcode_lst]
# 	for icdcode in icdcode_lst:
# 		try:
# 			ccs = icd2ccs[icdcode]
# 			description = ccscode2description[ccs].lower() 	
# 			if 'infect' in description:
# 				return True 
# 		except:
# 			pass
# 	return False 



# def filter_disorder(row):
# 	icdcode_text = row[6]
# 	icdcode_lst2 = icdcode_text_2_lst_of_lst(icdcode_text)
# 	icdcode_lst = reduce(lambda x,y:x+y, icdcode_lst2)
# 	icdcode_lst = [i.replace('.', '') for i in icdcode_lst]
# 	for icdcode in icdcode_lst:
# 		try:
# 			ccs = icd2ccs[icdcode]
# 			description = ccscode2description[ccs].lower() 	
# 			if 'disorder' in description:
# 				return True 
# 		except:
# 			pass 
# 	return False 

def filter_nervous(row):
	icdcode_lst = row2icdcodelst(row)
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'NVS':
				return True 
		except:
			pass 
	return False 

def filter_cancer(row):
	icdcode_lst = row2icdcodelst(row)
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'NEO':
				return True 
		except:
			pass 
	return False 



# def filter_cancer(row):
# 	icdcode_text = row[6]
# 	if 'cancer' in icdcode_text.lower() or 'neoplasm' in icdcode_text.lower() \
# 		or 'oncology' in icdcode_text.lower() or 'tumor' in icdcode_text.lower():
# 		return True 
# 	icdcode_lst2 = icdcode_text_2_lst_of_lst(icdcode_text)
# 	icdcode_lst = reduce(lambda x,y:x+y, icdcode_lst2)
# 	icdcode_lst = [i.replace('.', '') for i in icdcode_lst]
# 	for icdcode in icdcode_lst:
# 		try:
# 			ccs = icd2ccs[icdcode]
# 			description = ccscode2description[ccs].lower() 	
# 			if 'cancer' in description or 'neoplasm' in description or 'oncology' in description or 'tumor' in description:
# 				return True 
# 		except:
# 			pass 
# 	return False 

def filter_infect(row):
	icdcode_lst = row2icdcodelst(row)
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'INF':
				return True 
		except:
			pass 
	return False 


def filter_respiratory(row):
	icdcode_lst = row2icdcodelst(row)
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'RSP':
				return True 
		except:
			pass 
	return False 

def filter_digest(row):
	icdcode_lst = row2icdcodelst(row)
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'DIG':
				return True 
		except:
			pass 
	return False 




def write_row_to_csvfile(rows, fieldname, output_file):
	with open(output_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()	
		for row in rows:
			dic = {k:row[i] for i,k in enumerate(fieldname)}
			writer.writerow(dic)
	return	


nctid2year = dict() 
with open('data/nctid_date.txt', 'r') as fin:
	lines = fin.readlines()
for line in lines:
	nctid, start_year, completion_year = line.strip('\n').split('\t')
	start_year = 0 if start_year=='' else int(start_year.split()[-1])
	completion_year = 0 if completion_year == '' else int(completion_year.split()[-1])
	nctid2year[nctid] = start_year, completion_year  #### 0, 2018

def row2year(row):
	nctid = row[0]
	start_year, completion_year = nctid2year[nctid]
	return start_year, completion_year 


def select_and_split_data(input_file, filter_func, output_file_name):
	rows = csvfile2rows(input_file)
	rows = list(filter(filter_func, rows))
	fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 
				 'diseases', 'icdcodes', 'drugs', 'smiless', 'criteria', 'lead_sponsor', 'collaborator']
	write_row_to_csvfile(rows, fieldname, output_file_name)
	return


def smiles_txt_to_lst(text):
	"""
		"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
	"""
	text = text[1:-1]
	lst = [i.strip()[1:-1] for i in text.split(',')]
	return lst 

from copy import deepcopy

# nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria,lead_sponsor,collaborator

def clean_data(input_file, clean_file):
	"""
		remove placebo 
	"""
	rows = csvfile2rows(input_file)
	newrows = []
	fieldname = ['nctid','status','why_stop','label','phase','diseases','icdcodes','drugs','smiless','criteria','lead_sponsor','collaborator']
	with open(clean_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldname)
		writer.writeheader()	
		for row in rows:
			# drugs = row[7]
			# if 'placebo' not in drugs.lower():
			# 	newrows.append(row)
			# 	continue 
			# smiless = row[8]
			# newdrug, newsmiles = [], []
			# # assert len(smiles_txt_to_lst(drugs)) == len(smiles_txt_to_lst(smiless))
			# for drug, smiles in zip(smiles_txt_to_lst(drugs), smiles_txt_to_lst(smiless)):
			# 	if 'placebo' not in drug.lower():
			# 		newdrug.append(drug)
			# 		newsmiles.append(smiles)
			# 	else:
			# 		print(smiles)
			# newdrug = str(newdrug)
			# newsmiles = str(smiles)
			# assert len(newdrug) > 0

			# smiless = row[8]
			# if '[O--].[Mg++]' in smiless:
			# 	smiles_lst = smiles_txt_to_lst(smiless)
			# 	smiles_lst = set(smiles_lst)
			# 	smiles_lst.remove('[O--].[Mg++]')
			# 	if len(smiles_lst)==0:
			# 		continue 
			# 	smiles_lst = str(list(smiles_lst))
			# 	newrow = row[:8] + [smiles_lst] + row[9:]
			# else:
			# 	newrow = row 
			newrow = row 

			dic = {k:newrow[i] for i,k in enumerate(fieldname)}
			writer.writerow(dic)
	return	



if __name__ == "__main__":
	input_file = 'data/ongoing_data.csv'
	clean_file = "data/clean_ongoing_data.csv"

	clean_data(input_file, clean_file)
	#### remove placebo 

	print("------------ phase I -------------")
	select_and_split_data(clean_file, filter_phase_I, 'data/ongoing_phase_I.csv')
	print("----------- phase II -------------")
	select_and_split_data(clean_file, filter_phase_II, 'data/ongoing_phase_II.csv')
	print("----------- phase III ----------")
	select_and_split_data(clean_file, filter_phase_III, 'data/ongoing_phase_III.csv')



'''
origin 
	phase I
		pos = 997   neg = 558
	phase II
		pos = 1069   neg = 2377
	phase III
		pos = 2399   neg = 1458
	indication
		pos = 3146   neg = 2146





'''








