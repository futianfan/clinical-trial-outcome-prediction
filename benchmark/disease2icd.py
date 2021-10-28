'''
input:  

	"data/drugbank_trial_conditions.csv"

process:
	disease maps to icd? code 

output: 

	"data/disease2icd.pkl" 



'''






import csv, pickle 
from collections import defaultdict 
def disease2icd_func():
	file = "data/drugbank_trial_conditions.csv"
	with open(file, 'r') as csvfile:
		reader = list(csv.reader(csvfile, delimiter = ','))[1:]
	disease2icdcode = defaultdict(set)
	disease2icdcode2 = dict()
	for row in reader:
		diseasename1 = row[2].lower()
		diseasename2 = row[6].lower()
		icd10code = row[8]
		if icd10code.strip() == '':
			continue 
		disease2icdcode[diseasename1].add(icd10code)
		disease2icdcode[diseasename2].add(icd10code)
	for disease, icdcode in disease2icdcode.items():
		assert len(icdcode)==1 
		disease2icdcode2[disease] = list(icdcode)[0]
	return disease2icdcode2 


### disease -> icd code
if __name__ == "__main__":
	disease2icdcode = disease2icd_func()
	pickle.dump(disease2icdcode, open("data/disease2icd.pkl", 'wb')) 
	for disease, icd in disease2icdcode.items():
		if len(disease.split())==1:
			print(disease, icd)








