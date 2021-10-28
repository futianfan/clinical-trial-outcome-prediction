'''
input:  270k
	data/raw_data.csv

process:
	1. print statistics


output:
	None 

'''


import csv 
from collections import defaultdict
text2cnt = defaultdict(int)

raw_data_file = "data/raw_data.csv" 
with open(raw_data_file, 'r') as csvfile:
	reader = list(csv.reader(csvfile, delimiter = ','))[1:]
fieldname = ['nctid', 'status', 'why_stop', 'label', 'phase', 'diseases', 'drugs', 'title', 'criteria', 'summary']



drop_set = ['Active, not recruiting', 'Enrolling by invitation', 'No longer available',  
			'Not yet recruiting', 'Recruiting', 'Temporarily not available', 'Unknown status']


for row in reader:
	status = row[1]
	why_stop = row[2]
	label = row[3]
	phase = row[4]

	text2cnt[status] += 1

	# if status == 'Suspended':
	# 	text2cnt[label] += 1


	# text2cnt[status] += 1

	# if status not in drop_set:		
	# 	text = '\t'.join(row[3:4])
	# 	text2cnt[text] += 1

text2cnt_lst = sorted([(k,v) for k,v in text2cnt.items()], key = lambda x:x[1], reverse = True)

for k,v in text2cnt_lst[:]:
	print(k.strip(), v)

'''
Observation

	status

		Completed 150900
					-1 139332
					1 6534
					0 5034

		Recruiting 38492    
					-1 38489
					1 3		

		Unknown status 28093
					-1 28076
					1 13
					0 4

		Terminated 17270
					-1 16468
					0 589
					1 213

		Active, not recruiting 14236
					-1 13852
					1 234
					0 150

		Not yet recruiting 13331
					-1 13331

		Withdrawn 7355
					-1 7355

		Enrolling by invitation 2066
					-1 2066

		Suspended 1601
					-1 1600
					0 1



	why_stop

		'' 155449
		Slow accrual 173
		Lack of funding 140
		See termination reason in detailed description. 118
		low accrual 103
		slow accrual 101
		xxxxxxx 
		xxxxxxx
		xxxxxxx

	label

		-1 164755
		1 6747
		0 5624


	phase

		N/A 67084
		Phase 2 31069
		Phase 1 25649
		Phase 3 23084
		Phase 4 18714
		Phase 1/Phase 2 6520
		Phase 2/Phase 3 3347
		Early Phase 1 1658


'''







