'''
## drug maps to smiles
## input:  "data/drugbank_drugs_info.csv"
## output:  "data/drug2smiles.pkl"

'''



import csv, pickle 
from collections import defaultdict 
def drug2smiles_func():
	file = "data/drugbank_drugs_info.csv"
	with open(file, 'r') as csvfile:
		reader = list(csv.reader(csvfile, delimiter = ','))[1:]
	drug2smiles = defaultdict(set)
	drug2smiles2 = dict()
	for row in reader:
		smiles = row[27]
		if smiles.strip()=='':
			continue 
		drug1 = row[3].lower()
		drug2 = row[11].lower()
		drug2smiles[drug1].add(smiles)
		drug2smiles[drug2].add(smiles)
	for drug, smiles in drug2smiles.items():
		smiles = list(smiles)[0]
		drug2smiles2[drug] = smiles 
	#### to improve 
	'''
		 7: 53, 3: 1452, 1: 26851, 5: 178, 10: 16, 14: 6, 2: 6129, 
		 4: 504, 17: 8, 12: 8, 8: 38, 6: 83, 11: 12, 9: 17, 161: 1, 
		 21: 2, 15: 4, 32: 2, 13: 2, 31: 1, 22: 2, 23: 3, 16: 1, 18: 2, 104: 1, 19: 2
	'''
	return drug2smiles2

### disease -> icd code
if __name__ == "__main__":
	drug2smiles = drug2smiles_func()
	drug2smiles_file = "data/drug2smiles.pkl"
	pickle.dump(drug2smiles, open(drug2smiles_file, 'wb'))


'''
[
 0: 'id', 
 1: 'trial_id', 
 2: 'kind', 
 3: 'title', 
 4: 'description', 
 5: 'id', 
 6: 'intervention_id', 
 7: 'drug_id', 
 8: 'id', 
 9: 'type', 
 10: 'drugbank_id', 
 11: 'name', 
 12: 'state',   ----- solid liquid 
 13: 'description', 
 14: 'cas_number', 
 15: 'protein_formula', 
 16: 'protein_weight', 
 17: 'investigational', 
 18: 'approved', 
 19: 'vet_approved', 
 20: 'experimental', 
 21: 'nutraceutical', 
 22: 'illicit', 
 23: 'withdrawn', 
 'moldb_mono_mass', 
 'moldb_inchi', 
 'moldb_inchikey', 
 'moldb_smiles', 
 'moldb_average_mass', 
 'moldb_formula', 
 'synthesis_patent_id', 
 'protein_weight_details', 
 'biotech_kind']

'''
