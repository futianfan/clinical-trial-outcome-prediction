'''

- data/ongoing_predict_phase_I.txt
- data/ongoing_predict_phase_II.txt
- data/ongoing_predict_phase_III.txt


- data/ongoing_phase_I.csv
- data/ongoing_phase_II.csv
- data/ongoing_phase_III.csv

'''

import pandas as pd 
from pandas import DataFrame
import csv 
from tqdm import tqdm 
from xml.etree import ElementTree as ET 
from collections import defaultdict 
nctid2predict = dict()
nctid2sponsor = dict() 
nctid2info = dict()
sponsor2nctid_pred = defaultdict(lambda: [])
sponsor2top3 = []
for base_name in ['phase_I', 'phase_II', 'phase_III']:
	prediction_file = 'data/ongoing_predict_' + base_name + '.txt'
	with open(prediction_file, 'r') as fin:
		lines = fin.readlines()
		for line in lines:
			nctid, predict = line.strip().split() 
			nctid2predict[nctid] = float(predict)

	prediction_file = 'data/test_predict_' + base_name + '.txt'
	with open(prediction_file, 'r') as fin:
		lines = fin.readlines()
		for line in lines:
			nctid, predict = line.strip().split() 
			nctid2predict[nctid] = float(predict)


# nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria,lead_sponsor,collaborator 
for base_name in ['phase_I', 'phase_II', 'phase_III']:
	data_file = 'data/ongoing_' + base_name + '.csv'
	with open(data_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))[1:]
		for row in rows:
			nctid = row[0]
			sponsor = row[-2]
			phase = row[4]
			diseases = row[5]
			drugs = row[7]
			sponsor2nctid_pred[sponsor].append([nctid, nctid2predict[nctid]])
			nctid2info[nctid] = [phase, diseases, drugs]

	data_file = 'data/' + base_name + '_test.csv'
	with open(data_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))[1:]
		for row in tqdm(rows):
			nctid = row[0]
			xml_file = 'ctgov/' + nctid[:7] + 'xxxx/' + nctid + '.xml' 
			tree = ET.parse(xml_file)
			root = tree.getroot()	
			sponsor = root.find('sponsors').find('lead_sponsor').find('agency').text 
			phase = row[4]
			diseases = row[5]
			drugs = row[7]
			sponsor2nctid_pred[sponsor].append([nctid, nctid2predict[nctid]])
			nctid2info[nctid] = [phase, diseases, drugs]

# print(len(nctid2info), len(nctid2predict))
# exit()

month2num = ['January','February','March','April','May','June','July','August','September','October','November','December']
month2num = {v.lower():k for k,v in enumerate(month2num)}


def date2num(datestring):
	month = datestring.split()[0].lower() 
	month = month2num[month]

	if ',' in datestring:
		day = int(datestring.split(',')[0].split()[-1])
		year = int(datestring.split(',')[-1].strip())
		return month, day, year 
	day = 1 
	year = int(datestring.split()[-1])
	return month, day, year 

def nctid_2_date(nctid):
	xml_file = 'ctgov/' + nctid[:7] + 'xxxx/' + nctid + '.xml' 	
	tree = ET.parse(xml_file)
	root = tree.getroot()
	try:
		start_date = root.find('start_date').text	
		# start_date = int(start_date.split()[-1])
	except:
		start_date = ''
	try:
		completion_date = root.find('primary_completion_date').text
	except:
		try:
			completion_date = root.find('completion_date').text 
		except:
			completion_date = ''
	duration = 'unknown'
	if start_date != '' and completion_date != '':
		start_month, start_day, start_year = date2num(start_date)
		completion_month, completion_day, completion_year = date2num(completion_date)
		duration = (completion_year- start_year) * 365 + (completion_month - start_month) * 30 + completion_day - start_day 
		duration = str(duration)
	return start_date, completion_date, duration


for sponsor, nctid_pred_lst in sponsor2nctid_pred.items():
	nctid_pred_lst.sort(key=lambda x:x[1], reverse = True)
	top3score = sum([i[1] for i in nctid_pred_lst[:3]]) 
	sponsor2top3.append((sponsor, top3score))



def nctid2label_dict():
	nctid2outcome = dict() 
	outcome2label = dict() 
	nctid2label = dict() 
	with open("trialtrove/outcome2label.txt", 'r') as fin: 
		lines = fin.readlines() 
		for line in lines:
			outcome = line.split('\t')[0]
			label = line.strip().split('\t')[1]
			outcome2label[outcome] = label 

	with open("trialtrove/trial_outcomes_v1.csv", 'r') as csvfile: 
		csvreader = list(csv.reader(csvfile))[1:]
		for row in csvreader:
			nctid = row[0]
			outcome = row[1]
			nctid2outcome[nctid] = outcome 

	for nctid,outcome in nctid2outcome.items():
		nctid2label[nctid] = outcome2label[outcome]

	return nctid2label 

nctid2label = nctid2label_dict() 
# print(nctid2label)



file_out = 'sponsor_info.xls'
data_lst = []


with open('sponsor_info.txt', 'w') as fout:
	columns_header = ['sponsor', 'nctid', 'phase', 'disease', 'drug', 'prediction', 'groundtruth']
	fout.write('\t'.join(columns_header) + '\n')
	for sponsor, nctid_pred_lst in tqdm(sponsor2nctid_pred.items()):
		nctid_pred_lst.sort(key=lambda x:x[1], reverse = True)
		for nctid, pred in nctid_pred_lst:
			phase, diseases, drugs = nctid2info[nctid]
			label = 'unknown'
			if nctid in nctid2label and nctid2label[nctid]!=-1:
				label = str(nctid2label[nctid])
				label = label.strip() 
				if label == '-1':
					label = '0' 
			start_date, completion_date, duration = nctid_2_date(nctid)		
			columns = [sponsor, nctid, phase, diseases, drugs, str(pred)[:5], label, start_date, completion_date, duration]
			fout.write('\t'.join(columns) + '\n')
			data_lst.append(columns)

nct2allfeature = dict()
for sponsor, nctid, phase, diseases, drug, prediction, groundtruth, start_date, completion_date, duration in data_lst:
	nct2allfeature[nctid] = sponsor, phase, diseases, drug, prediction, groundtruth, start_date, completion_date, duration  

data_lst = []
for nctid, (sponsor, phase, diseases, drug, prediction, groundtruth, start_date, completion_date, duration) in nct2allfeature.items():
	data_lst.append([sponsor, nctid, phase, diseases, drug, prediction, groundtruth, start_date, completion_date, duration])

print('# of data points', len(data_lst))
data = {
	'sponsor': [data[0] for data in data_lst], 
	'nctid': [data[1] for data in data_lst], 
	'phase': [data[2] for data in data_lst], 
	'diseases': [data[3] for data in data_lst], 
	'drug': [data[4] for data in data_lst], 
	'prediction': [data[5] for data in data_lst], 
	'groundtruth': [data[6] for data in data_lst], 
	'start_date': [data[7] for data in data_lst], 
	'completion_date': [data[8] for data in data_lst], 
	'duration': [data[9] for data in data_lst], 
}

df = DataFrame(data)
print(df.shape)
df.to_excel(file_out)



sponsor2top3.sort(key = lambda x:x[1], reverse = True)
for j in [i[0] for i in sponsor2top3[:10]]:
	print(j)






"""

google sheet 

adding some trials that we already know results

Can you also train another model for predicting trial duration


"""



