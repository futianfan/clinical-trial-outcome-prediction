import matplotlib.pyplot as plt 
import numpy as np 
data_file = "data/raw_data.csv"
import csv, os, pickle  
from tqdm import tqdm 
import numpy as np 
from functools import reduce 
from xml.etree import ElementTree as ET
raw_folder = "raw_data"


import seaborn as sns
from matplotlib import font_manager

font_dirs = ["./"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
sns.set(rc={'figure.figsize':(6,6)})
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font = "Helvetica", font_scale=1.5)




pattern_string = "participants group_id"

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

def xmlfile_2_startyear(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	try:
		start_date = root.find('start_date').text	
		start_date = int(start_date.split()[-1])
	except:
		start_date = -1
	return start_date

def file2patientnumber(xml_file):
	os.system('grep "' + pattern_string + '" ' + xml_file + '  > tmp')
	with open("tmp", 'r') as fin:
		lines = fin.readlines()
	summ = 0
	for line in lines:
		summ += int(line.split('"')[3])
	return summ  


#### data/all_xml
# if True:
# 	year_lst = []
# 	nctid2year = dict()
# 	nctid2patientnumber = dict() 
# 	with open("data/all_xml") as fin:
# 		lines = fin.readlines() 
# 		for line in tqdm(lines):
# 			file = line.strip()
# 			nctid = line.strip().split('/')[-1].split('.')[0]
# 			start_year = xmlfile_2_startyear(file)
# 			try:
# 				patientnumber = file2patientnumber(file)
# 				nctid2patientnumber[nctid] = patientnumber
# 				print(patientnumber)
# 			except:
# 				pass 
# 			if start_year != -1:
# 				year_lst.append(start_year)
# 				nctid2year[nctid] = start_year 


# 	pickle.dump(year_lst, open("data/year_histogram.pkl", 'wb'))
# 	data = year_lst ##### [2008, 2007, 2000, 2006, 2007, 2008, 2000, 1999, .......]
# 	data = list(filter(lambda x:x>1998 and x<2020, data))
# 	pickle.dump(nctid2year, open("data/nctid2year_all.pkl", 'wb'))
# 	pickle.dump(nctid2patientnumber, open("data/all_nctid2patientnumber.pkl", 'wb'))

# 	plt.cla()
# 	fig, ax = plt.subplots()
# 	num_bins = 23
# 	n, bins, patches = ax.hist(data, num_bins, )
# 	plt.tick_params(labelsize=15)
# 	ax.set_xlabel('Year', fontsize = 25)  
# 	ax.set_ylabel('Number of selected trials', fontsize = 24)  
# 	plt.tight_layout() 
# 	# ax.set_title(r'Histogram of trial number in each year') 
# 	# fig.set_facecolor('cyan')  #
# 	plt.savefig("figure/all_histogram.png")
# 	plt.cla()



import seaborn as sns
from matplotlib import font_manager

font_dirs = ["./"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

fig, axes = plt.subplots(1,3, figsize=(25,6))

sns.set(rc={'figure.figsize':(6,6)})
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font = "Helvetica", font_scale=1.5)




ax = axes[0]

# if not (os.path.exists("data/nctid2year.pkl") and os.path.exists("data/nctid2patientnumber.pkl")):
if True:
	year_lst = []
	nctid2year = dict()
	nctid2patientnumber = dict() 
	with open(data_file) as f:
		reader = list(csv.reader(f))[1:]
		for line in tqdm(reader):
			nctid = line[0]
			file = os.path.join(raw_folder, nctid[:7]+"xxxx/"+nctid+".xml")
			# assert os.path.exists(file)
			start_year = xmlfile_2_startyear(file)
			try:
				patientnumber = file2patientnumber(file)
				nctid2patientnumber[nctid] = patientnumber
				# print(patientnumber)
			except:
				pass 
			if start_year != -1:
				year_lst.append(start_year)
				nctid2year[nctid] = start_year 


	pickle.dump(year_lst, open("data/year_histogram.pkl", 'wb'))
	data = year_lst ##### [2008, 2007, 2000, 2006, 2007, 2008, 2000, 1999, .......]
	data = list(filter(lambda x:x>1998, data))
	pickle.dump(nctid2year, open("data/nctid2year.pkl", 'wb'))
	pickle.dump(nctid2patientnumber, open("data/nctid2patientnumber.pkl", 'wb'))

	# plt.cla()
	# fig, ax = plt.subplots()
	ax = axes[0]
	num_bins = 23
	n, bins, patches = ax.hist(data, num_bins, )
	plt.tick_params(labelsize=15)
	ax.set_xlabel('Year', fontsize = 25)  
	ax.set_ylabel('Number of selected trials', fontsize = 24)  
	ax.set_title('A', fontsize = 25)

	# plt.tight_layout() 
	# ax.set_title(r'Histogram of trial number in each year') 
	# fig.set_facecolor('cyan')  #
	# plt.savefig("histogram.png")
	# plt.cla()

# else:
# 	nctid2year = pickle.load(open("data/nctid2year.pkl", 'rb'))
# 	nctid2patientnumber = pickle.load(open("data/nctid2patientnumber.pkl", 'rb'))



ax = axes[1]
# if not os.path.exists("data/nctid2label.pkl"):
if True:
	nctid2label = dict() 
	nctid2drug = dict() 
	nctid2disease = dict() 
	nctid2icd = dict() 
	with open("data/raw_data.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			label = int(row[3])
			nctid2label[nctid] = label
			drug = row[7].strip('"[],')
			drug_lst = drug.strip("'").split("', '")
			# print("drug", drug_lst)
			nctid2drug[nctid] = drug_lst
			disease = row[5].strip('"[],')
			disease_lst = disease.strip("'").split("', '")
			# print("disease", disease_lst)
			nctid2disease[nctid] = disease_lst 	
			icdcode_lst = row2icdcodelst(row)
			nctid2icd[nctid] = icdcode_lst 

		pickle.dump(nctid2label, open("data/nctid2label.pkl", 'wb'))
		pickle.dump(nctid2drug, open("data/nctid2drug.pkl", 'wb'))
		pickle.dump(nctid2disease, open("data/nctid2disease.pkl", 'wb'))
		pickle.dump(nctid2icd, open("data/nctid2icd.pkl", 'wb'))
else:
	nctid2label = pickle.load(open("data/nctid2label.pkl", 'rb'))
	nctid2drug = pickle.load(open("data/nctid2drug.pkl", 'rb'))
	nctid2disease = pickle.load(open("data/nctid2disease.pkl", 'rb'))
	nctid2icd = pickle.load(open("data/nctid2icd.pkl", 'rb'))

disease_lst = [disease for nctid, disease in nctid2disease.items()]
disease_lst = list(reduce(lambda x,y:x+y, disease_lst))
print("total disease", len(set(disease_lst)))
drug_lst = [drug for nctid, drug in nctid2drug.items()]
drug_lst = list(reduce(lambda x,y:x+y, drug_lst))
print("total drug", len(set(drug_lst)))


##### year vs % of approval 
from collections import defaultdict 
year2num = defaultdict(lambda:[0,0])
for nctid, year in nctid2year.items():
	label = nctid2label[nctid]
	year2num[year][0] += label 
	year2num[year][1] += 1 

year2approvalrate = []
for year in range(1998,2021):
	year2approvalrate.append(year2num[year][0] / year2num[year][1] * 100)
pickle.dump(year2approvalrate, open("data/year2approvalrate.pkl", 'wb'))
ax.plot(list(range(1998,2021)),year2approvalrate)
ax.set_xlabel("Year", fontsize = 24)
ax.set_ylabel("Success rate (%)", fontsize=25)
ax.set_title('B', fontsize = 25)
# plt.tight_layout() 
# plt.savefig("year2approvalrate.png")
# plt.cla() 







##### year vs # of recruit
ax = axes[2]
year2recruitnum = defaultdict(lambda:[0,0])
for nctid, patientnumber in nctid2patientnumber.items():
	try:
		year = nctid2year[nctid]
	except:
		continue 
	year2recruitnum[year][0] += patientnumber 
	year2recruitnum[year][1] += 1 
year2recruitnum_lst = []
for year in range(1998, 2021): 
	year2recruitnum_lst.append(year2recruitnum[year][0])

pickle.dump(year2recruitnum_lst, open("data/year2recruitnum.pkl", 'wb'))
ax.plot(list(range(1998,2021)),year2recruitnum_lst)
ax.set_xlabel("Year", fontsize = 24)
ax.set_ylabel("# of recruited patients", fontsize=25)
ax.set_title('C', fontsize = 25)


# plt.tight_layout() 
# plt.savefig("year2recruitedpatients.png")
plt.savefig('figure/1.pdf', bbox_inches='tight')
# plt.savefig("time_distribution.pdf")
exit()






















print('-----------------------------------------------')
data = [year for nctid,year in nctid2year.items()]
target_year_range = [(1900,1999), (2000,2004), (2005,2009), (2010,2014), (2015,2022)]
for year1, year2 in target_year_range:
	print('-----------------------------------------------')
	print(year1, year2)
	selected_nctids = [nctid for nctid,year in nctid2year.items() if year>=year1 and year<=year2]
	positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
	print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
	patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
	print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
	disease_set, drug_set = set(), set() 
	for nctid in selected_nctids:
		disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
		disease_set = disease_set.union(set(disease_lst))
		drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
		drug_set = drug_set.union(set(drug_lst))
	print("disease ", len(disease_set))
	print("drug", len(drug_set))


print("# trials", len(selected_nctids))

print('-----------------------------------------------')
print('########### neoplasm ############')
selected_nctids = [nctid for nctid,disease in nctid2disease.items() \
						if 'neoplasm' in ' '.join(disease).lower() or \
						'tumor' in ' '.join(disease).lower() or \
						'cancer' in ' '.join(disease).lower()]
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set))


from ccs_utils import file2_icd2ccs_and_ccs2description, file2_icd2ccsr
# icd2ccs, ccscode2description = file2_icd2ccs_and_ccs2description() 
icd2ccsr = file2_icd2ccsr()

print('-----------------------------------------------')
print('########### respiratory ############')
selected_nctids = []
for nctid, icdcode_lst in nctid2icd.items():
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'RSP':
				selected_nctids.append(nctid)
				break 
		except:
			pass 
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 



print('-----------------------------------------------')
print('########### digestive ############')
selected_nctids = []
for nctid, icdcode_lst in nctid2icd.items():
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'DIG':
				selected_nctids.append(nctid)
				break 
		except:
			pass 
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 




print('-----------------------------------------------')
print('########### nervous ############')
selected_nctids = []
for nctid, icdcode_lst in nctid2icd.items():
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if ccsr == 'NVS':
				selected_nctids.append(nctid)
				break 
		except:
			pass 
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 


print('-----------------------------------------------')
print('########### other diseases ############')
selected_nctids = []
for nctid, icdcode_lst in nctid2icd.items():
	for icdcode in icdcode_lst:
		try:
			ccsr = icd2ccsr[icdcode]
			if not (ccsr == 'NVS' or ccsr == 'DIG' or ccsr == 'RSP' or ccsr == 'NEO'):
				selected_nctids.append(nctid)
				break 
		except:
			pass 
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 





print('-----------------------------------------------')
print('########### phase I ############')
selected_nctids = []
if True:
	with open("data/phase_I_train.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_I_test.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_I_valid.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)

phase1_nctids = selected_nctids[:]
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 









print('-----------------------------------------------')
print('########### phase II ############')
selected_nctids = []
if True:
	with open("data/phase_II_train.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_II_test.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_II_valid.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)

phase2_nctids = selected_nctids[:]

positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 






print('-----------------------------------------------')
print('########### phase III ############')
selected_nctids = []
if True:
	with open("data/phase_III_train.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_III_test.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)
	with open("data/phase_III_valid.csv") as fin:
		readers = list(csv.reader(fin))[1:]
		for row in readers:
			nctid = row[0]
			selected_nctids.append(nctid)

phase3_nctids = selected_nctids[:]
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 


print('-----------------------------------------------')
print('########### phase 1,2,3 ############')
selected_nctids = phase1_nctids + phase2_nctids + phase3_nctids 
positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
disease_set, drug_set = set(), set()  
for nctid in selected_nctids:
	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
	disease_set = disease_set.union(set(disease_lst))
	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
	drug_set = drug_set.union(set(drug_lst))
print("disease ", len(disease_set))
print("drug", len(drug_set)) 





# print('-----------------------------------------------')
# print('########### indication ############')
# selected_nctids = []
# if True:
# 	with open("data/indication_train.csv") as fin:
# 		readers = list(csv.reader(fin))[1:]
# 		for row in readers:
# 			nctid = row[0]
# 			selected_nctids.append(nctid)
# 	with open("data/indication_test.csv") as fin:
# 		readers = list(csv.reader(fin))[1:]
# 		for row in readers:
# 			nctid = row[0]
# 			selected_nctids.append(nctid)
# 	with open("data/indication_valid.csv") as fin:
# 		readers = list(csv.reader(fin))[1:]
# 		for row in readers:
# 			nctid = row[0]
# 			selected_nctids.append(nctid)

# positive_sample = len(list(filter(lambda x:x in nctid2label and nctid2label[x]==1, selected_nctids)))
# print("total samples:", len(selected_nctids), "positive sample:", positive_sample, "negative sample", len(selected_nctids)-positive_sample)
# patientnumber_lst = [nctid2patientnumber[nctid] for nctid in selected_nctids if nctid in nctid2patientnumber]
# print("patient number ", np.mean(patientnumber_lst), np.std(patientnumber_lst), np.percentile(patientnumber_lst,[25,50,75]))
# disease_set, drug_set = set(), set()  
# for nctid in selected_nctids:
# 	disease_lst = nctid2disease[nctid] if nctid in nctid2disease else []
# 	disease_set = disease_set.union(set(disease_lst))
# 	drug_lst = nctid2drug[nctid] if nctid in nctid2drug else []
# 	drug_set = drug_set.union(set(drug_lst))
# print("disease ", len(disease_set))
# print("drug", len(drug_set)) 



