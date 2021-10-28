'''
icd code maps to description 
input: "icdcode/ICD-10-Medical-Diagnosis-Codes.pdf"
output:  "icdcode/description2icd10.pkl" 

'''



import icd10, PyPDF2, pickle, os, csv 

# def code2description(code):
# 	# code = "A1803"
# 	code = icd10.find(code)
# 	description = code.description 
# 	return description 

# def extract_icdcode(file, pkl_file):
# 	f = open(file, 'rb')
# 	description2icd = dict()
# 	fileReader = PyPDF2.PdfFileReader(file)
# 	for pagenum in range(fileReader.numPages):
# 		pageObj = fileReader.getPage(pagenum)
# 		text = pageObj.extractText()
# 		text = text.split('\n')
# 		text = list(filter(lambda x:len(x.strip())>0, text))
# 		if pagenum == 0:
# 			text = list(filter(lambda x:x[0]=='A', text))
# 			print(text)
# 		for i in text:
# 			icd = i.split()[0]
# 			description = ' '.join(i.strip().split()[1:]).lower()
# 			description2icd[description] = icd 
# 			# print(description, icd)
# 	print("code number is", str(len(description2icd)))
# 	# code number is 76331
# 	pickle.dump(description2icd, open(pkl_file, 'wb'))



###  csv_file = 'icdcode/icd_10_direct_mapping.csv'
def extract_icdcode(csv_file, pkl_file):
	with open(csv_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))
	description2icd = dict() 
	for row in rows:
		icd = row[0]
		description = row[1]
		print(icd)
		description2icd[description] = icd
	print("code number is", str(len(description2icd)))	
	pickle.dump(description2icd, open(pkl_file, 'wb'))




if __name__ == "__main__":
	# file = "icdcode/ICD-10-Medical-Diagnosis-Codes.pdf"
	csv_file = 'icdcode/icd_10_direct_mapping.csv'
	pkl_file = "icdcode/description2icd10.pkl"
	if not os.path.exists(pkl_file):
		extract_icdcode(csv_file, pkl_file)
	description2icd10 = pickle.load(open(pkl_file, 'rb'))
	# for description, icd in description2icd10.items():
	# 	#if len(description.split())==1:
	# 	print(description, "  -->  ", icd)






