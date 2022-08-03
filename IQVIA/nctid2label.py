import csv 


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
print(nctid2label)

