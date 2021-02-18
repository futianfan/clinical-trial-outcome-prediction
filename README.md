# clinical-trial-outcome-prediction










## Raw Data 

```bash

ls ./ClinicalTrialGov  

```

It is downloaded from [ClinicalTrial.gov](https://clinicaltrials.gov/). 
It is 8.6+G, containing 348,891+ clinical trial records. 
The data size grows with time.  


































## Conda Environment

```bash

cd /project/molecular_data/graphnn/ctgov

conda activate predict_drug_clinical_trial
source activate predict_drug_clinical_trial 
```

Use ctgov.yml to setup conda environment. 








## Data Preprocess 




### (1) Collect all the NCTIDs. 348,891 IDs.
input: ClinicalTrialGov/   
output: all_xml
```bash
find ClinicalTrialGov/ -name NCT*.xml | sort > all_xml
```



### (2) diseaes -> icd10
input: ClinicalTrialGov/* & all_xml   
output:	ctgov_data/diseases.csv  
```bash 
python src/collect_disease_from_raw.py
```


### (3) drug -> SMILES 
input:iqvia_data/drugbank_drugs_info.csv   
output:iqvia_data/drug2smiles.pkl   
```bash
### need optimize 
python src/drug2smiles.py 
```




### (4) Aggregation

input:     
	1. ctgov_data/diseases.csv  
	2. iqvia_data/drug2smiles.pkl  
	3. all_xml        
output: ctgov_data/raw_data.csv
```bash
python src/collect_raw_data.py | tee process.log 
```
It takes around 20 minutes.   




## Dataset of clinical trial outcome prediction 



## Model 

```bash



```


## Contact

Please contact futianfan@gmail.com for help or submit an issue. 


