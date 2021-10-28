data_file = "data/raw_data.csv"
import csv, os, pickle  
from tqdm import tqdm 
from xml.etree import ElementTree as ET
raw_folder = "raw_data"


def xmlfile_2_startyear(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	try:
		start_date = root.find('start_date').text	
		start_date = int(start_date.split()[-1])
	except:
		start_date = -1
	return start_date


year_lst = []
with open(data_file) as f:
	reader = list(csv.reader(f))[1:]
	for line in tqdm(reader):
		nctid = line[0]
		file = os.path.join(raw_folder, nctid[:7]+"xxxx/"+nctid+".xml")
		assert os.path.exists(file)
		start_year = xmlfile_2_startyear(file)
		if start_year != -1:
			year_lst.append(start_year)




