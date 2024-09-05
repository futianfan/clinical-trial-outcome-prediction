# Env setup
# biobert-embedding install locally
python3 -m pip install -e biobert_embedding/



# Download Clinical Trial data
mkdir -p raw_data
cd raw_data
wget https://clinicaltrials.gov/AllPublicXML.zip
unzip AllPublicXML.zip
cd ../
find raw_data/ -name NCT*.xml | sort > data/all_xml


# Drugbank
# drug bank dat needs to be purchased from the DrugBank site. we have a copy in csv

# ClinicalTable  ICD10

# MoleculeNet - ADMET data


# DATA processsing
# 1. Clinical trial (disease info) ->  ICD10
# input: 	348k data  
	# 1. ClinicalTrialGov/NCTxxxx/xxxxxx.xml 
	# 2. all_xml
# Output: 'data/diseases.csv' 
python benchmark/collect_disease_from_raw.py

# 2. Clinical Trial (drug info) -> SMILE
## input:  "data/drugbank_drugs_info.csv"
## output:  "data/drug2smiles.pkl"
python benchmark/drug2smiles.py 

# 3. Filter clinical trials
# input: 	370K data  
# 	1. ClinicalTrialGov/NCTxxxx/xxxxxx.xml & all_xml    
# 	1. data/diseases.csv  
# 	2. data/drug2smiles.pkl          
# output: data/raw_data.csv
python benchmark/collect_raw_data.py | tee data_process.log 
python benchmark/nctid2date.py 

# 4. Split the phase of trials
# input: 	9k data ?  
# 	1. ctgov_data/raw_data.csv 

# nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria
# output:
# 	1. ctgov_data/phase_I.csv 
# 	2. ctgov_data/phase_II.csv 
# 	3. ctgov_data/phase_III.csv 
# 	4. ctgov_data/trial.csv 
python benchmark/data_split.py

# 5. ICD-10 code hierarchy
# input:
# 	data/raw_data.csv

# output: 
# 	data/icdcode2ancestor_dict.pkl (icdcode to its ancestors)
# 	icdcode_embedding 
python benchmark/icdcode_encode.py

# 6. Sentence embedding for clinical protocol
# input:
# 	data/raw_data.csv

# output:
# 	data/sentence2embedding.pkl (preprocessing)
# 	protocol_embedding 
python benchmark/protocol_encode.py
