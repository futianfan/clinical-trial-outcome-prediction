import pandas as pd 

'''
14 columns
ICD-10-CM Code	ICD-10-CM Code Definition	Beta Version CCS Category	Beta Version CCS Category Description	CCSR1	CCSR1 Label	CCSR2	CCSR2 Label	CCSR3	CCSR3 Label	CCSR4	CCSR4 Label	CCSR5	CCSR5 Label

'''
def rawfile2dict():
	'''
		ccs beta code
	'''
	file = "icdcode/DXCCSR-vs-Beta-CCS-Comparison.xlsx"
	icd2ccs_file = "icdcode/icd2ccs.txt"
	ccscode2description_file = 'icdcode/ccs2description.txt'
	contents = pd.read_excel(open(file, 'rb'), sheet_name = 'ICD-10-CM Code Detail')
	N, _ = contents.shape 

	icd10code_lst = contents['ICD-10-CM Code']
	ccs_beta_lst = contents['Beta Version CCS Category']
	ccs_description_lst = contents['Beta Version CCS Category Description']
	icd2ccs = dict()
	ccscode2description = dict() 
	for i in range(N):
		icdcode = icd10code_lst[i]
		ccscode = ccs_beta_lst[i]
		ccs_description = ccs_description_lst[i]
		icd2ccs[icdcode] = ccscode 
		ccscode2description[ccscode] = ccs_description
	with open(icd2ccs_file, 'w') as fout:
		for k,v in icd2ccs.items():
			fout.write(str(k) + '\t' + str(v) + '\n')
	with open(ccscode2description_file, 'w') as fout:
		for k,v in ccscode2description.items():
			fout.write(str(k) + '\t' + str(v) + '\n')
	return icd2ccs, ccscode2description 


def rawfile2dict_CCSR():
	file = "icdcode/DXCCSR-vs-Beta-CCS-Comparison.xlsx"
	icd2ccsr_file = "icdcode/icd2ccsr.txt"
	contents = pd.read_excel(open(file, 'rb'), sheet_name = 'ICD-10-CM Code Detail')
	N, _ = contents.shape 

	icd10code_lst = contents['ICD-10-CM Code']
	ccsr_lst = contents['CCSR1']
	icd2ccsr = dict()
	for i in range(N):
		icdcode = icd10code_lst[i]
		ccsr = ccsr_lst[i][:3]
		icd2ccsr[icdcode] = ccsr 
	with open(icd2ccsr_file, 'w') as fout:
		for k,v in icd2ccsr.items():
			fout.write(str(k) + '\t' + str(v) + '\n')
	return icd2ccsr 


def file2_icd2ccsr():
	icd2ccsr_file = "icdcode/icd2ccsr.txt"
	with open(icd2ccsr_file, 'r') as fin:
		lines = fin.readlines()	
	icd2ccsr = {line.split()[0]:line.split()[1] for line in lines}
	return icd2ccsr


def file2_icd2ccs_and_ccs2description():
	icd2ccs = dict()
	icd2ccs_file = "icdcode/icd2ccs.txt"
	ccscode2description_file = 'icdcode/ccs2description.txt'
	ccscode2description = dict() 
	with open(icd2ccs_file, 'r') as fin:
		lines = fin.readlines()
	icd2ccs = {line.split()[0]:line.split()[1] for line in lines}
	with open(ccscode2description_file, 'r') as fin:
		lines = fin.readlines() 
	ccscode2description = {line.split()[0]:line.split()[1] for line in lines}
	return icd2ccs, ccscode2description


# icd2ccs, ccscode2description = rawfile2dict() 


def cancer_filter_icd10code(icd10code):
	icd2ccs, ccscode2description = file2_icd2ccs_and_ccs2description() 
	ccs = icd2ccs[icd10code]
	description = ccscode2description[ccs]
	return 'cancer' in description.lower() 





if __name__ == "__main__":
	icd2ccsr = rawfile2dict_CCSR() 






















