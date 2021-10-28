import os.path as op
from xml.dom import minidom
from xml.etree import ElementTree as ET


folder = 'ClinicalTrialGov'

def nctid2fulltext(nctid):
	subfolder = nctid[:7]+'xxxx'
	file = op.join(folder, subfolder, nctid + '.xml')


xml_file = 'NCT01884350.xml'
'''
ClinicalTrialGov/NCT0188xxxx/NCT01884350.xml
'''

tree = ET.parse(xml_file)
root = tree.getroot() 









# def xml_file_2_tuple(xml_file):
# 	tree = ET.parse(xml_file)
# 	root = tree.getroot()
# 	nctid = root.find('id_info').find('nct_id').text	### nctid: 'NCT00000102'
# 	study_type = root.find('study_type').text 
# 	if study_type != 'Interventional':
# 		return (None,)  ### invalid 

# 	interventions = [i for i in root.findall('intervention')]
# 	drug_interventions = [i.find('intervention_name').text for i in interventions \
# 														if i.find('intervention_type').text=='Drug']
# 														# or i.find('intervention_type').text=='Biological']
# 	if len(drug_interventions)==0:
# 		return (None,)

# 	try:
# 		status = root.find('overall_status').text 
# 	except:
# 		status = ''
# 	# if status in drop_set:
# 	# 	return (None,)  ### invalid 
# 	try:
# 		why_stop = root.find('why_stopped').text
# 	except:
# 		why_stop = ''
# 	label = root2outcome(root)
# 	label = -1 if label is None else label 
# 	try:
# 		phase = root.find('phase').text 
# 		# print("phase\n\t\t", phase)
# 	except:
# 		phase = ''
# 	conditions = [i.text for i in root.findall('condition')]

# 	try:
# 		criteria = root.find('eligibility').find('criteria').find('textblock').text 
# 		# print("criteria\n\t\t", criteria)
# 	except:
# 		criteria = ''
# 	#if criteria != '':
# 	#	assert "Inclusion Criteria:" in criteria 
# 	#	assert "Exclusion Criteria:" in criteria 
# 	# title = root.find('brief_title').text	
# 	# try: 
# 	# 	summary = root.find('brief_summary').text 
# 	# 	# print("summary\n\t\t", summary)
# 	# except:
# 	# 	summary = '' 

# 	conditions = [i.lower() for i in conditions]
# 	drugs = [i.lower() for i in drug_interventions]

# 	return nctid, status.lower(), why_stop.lower(), label, phase.lower(), conditions, drugs, criteria
# 	# return nctid, status.lower(), why_stop.lower(), label, phase.lower(), conditions, drugs, title, criteria, summary





















