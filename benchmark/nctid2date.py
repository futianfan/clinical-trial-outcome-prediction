import csv, os
from tqdm import tqdm 
from xml.etree import ElementTree as ET


def xmlfile_2_date(xml_file):
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
	return start_date, completion_date 


raw_folder = "raw_data"
nctid_lst = []
total_num, start_num, completion_num = 0, 0, 0 
with open("data/raw_data.csv") as fin, open("data/nctid_date.txt", 'w') as fout:
	readers = list(csv.reader(fin))[1:]
	for row in tqdm(readers):
		nctid = row[0]
		file = os.path.join(raw_folder, nctid[:7]+"xxxx/"+nctid+".xml")
		start_date, completion_date = xmlfile_2_date(file)
		if start_date != '':
			start_num += 1
		if completion_date != '':
			completion_num += 1
		total_num += 1
		fout.write(nctid + '\t' + start_date + '\t' + completion_date + '\n')

print("total_num", total_num)
print("start_num", start_num)
print("completion_num", completion_num)


