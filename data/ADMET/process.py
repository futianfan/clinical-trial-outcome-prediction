import csv, random
random.seed(3)
valid_ratio = 0.2

def check_empty_smiles(smiles):
	if smiles.strip()=='':
		assert False

def lines_write_to_2_files(lines, prefix, valid_ratio):
	'''
		train & valid 
	'''
	N = len(lines)
	random.shuffle(lines)
	valid_num = int(valid_ratio * N)
	train_file = prefix + "_train.txt"
	valid_file = prefix + "_valid.txt"
	with open(train_file, 'w') as fo1, open(valid_file, 'w') as fo2:
		for line in lines[:-valid_num]:
			fo1.write(line)
		for line in lines[-valid_num:]:
			fo2.write(line)
	return 


### output: smiles \t 0/1 
input_file = "absorption/bioavailability.csv"
output_prefix = "cooked/absorption"
with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	lines = []
	for idx,row in enumerate(reader):
		if idx == 0:
			continue 
		smiles = row[-1]
		if smiles.strip()=='':
			continue
		label = 1 if row[2]=='1' else 0
		lines.append(smiles + '\t' + str(label) + '\n')
	lines_write_to_2_files(lines, output_prefix, valid_ratio)





input_file = "distribution/BBB.csv"
output_prefix = "cooked/distribution"
with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	lines = []
	for idx,row in enumerate(reader):
		if idx == 0:
			continue 
		smiles = row[1]
		if smiles.strip()=='':
			continue
		label = 1 if row[2]=='1' else 0
		lines.append(smiles + '\t' + str(label) + '\n')
	lines_write_to_2_files(lines, output_prefix, valid_ratio)




input_file = "metabolism/CYP2C19.csv"
output_prefix = "cooked/metabolism"
with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for idx,row in enumerate(reader):
		if idx == 0:
			continue 
		smiles = row[-2]
		if smiles.strip()=='':
			continue
		label = 1 if row[-1]=='1' else 0
		lines.append(smiles + '\t' + str(label) + '\n')
	lines_write_to_2_files(lines, output_prefix, valid_ratio)



input_file = "excretion/Clearance_eDrug3D.csv"
output_prefix = "cooked/excretion"
with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for idx,row in enumerate(reader):
		if idx == 0:
			continue 
		smiles = row[-1]
		if smiles.strip()=='':
			continue
		label = 1 if float(row[-2])<12 else 0
		lines.append(smiles + '\t' + str(label) + '\n')
	lines_write_to_2_files(lines, output_prefix, valid_ratio)







input_file = "toxicity/toxcast_data.csv"
output_prefix = "cooked/toxicity"
with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for idx,row in enumerate(reader):
		if idx == 0:
			continue 
		f = lambda x:True if len(x)>0 else False
		smiles = row[0]
		if smiles.strip()=='':
			continue
		label = list(filter(f,row[1:]))
		if len(label)<30:
			continue 
		label = [float(i) for i in label]
		if sum(label)==0:
			lines.append(smiles + '\t1\n')
		else:
			lines.append(smiles + '\t0\n')
	lines_write_to_2_files(lines, output_prefix, valid_ratio)









