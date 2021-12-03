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
with open("data/raw_data.csv") as fin, open("data/nctid_date.txt", 'w') as fout:
	readers = list(csv.reader(fin))[1:]
	for row in tqdm(readers):
		nctid = row[0]
		file = os.path.join(raw_folder, nctid[:7]+"xxxx/"+nctid+".xml")
		start_date, completion_date = xmlfile_2_date(file)
		fout.write(nctid + '\t' + start_date + '\t' + completion_date + '\n')




