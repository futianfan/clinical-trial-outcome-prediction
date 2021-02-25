# clinical-trial-outcome-prediction



## 1. Conda Environment

```bash

conda env create -f conda.yml


conda activate predict_drug_clinical_trial
```

The user can use conda.yml as reference to setup conda environment. 







































## 2. Raw Data 

```bash

ls ./ClinicalTrialGov  

```

It is downloaded from [ClinicalTrial.gov](https://clinicaltrials.gov/). 
It is 8.6+G, containing 348,891+ clinical trial records. 
The data size grows with time because more clinical trial records are added.  




































































## 3. Data Preprocess 


### 3.1 Collect all the 348,891 NCTIDs.
input: ClinicalTrialGov/   
output: data/all_xml 
```bash
find ClinicalTrialGov/ -name NCT*.xml | sort > data/all_xml
```


### 3.2 diseaes -> icd10
input: ClinicalTrialGov/* & data/all_xml   
output:	ctgov_data/diseases.csv  
```bash 
python src/collect_disease_from_raw.py
```


### 3.3 drug -> SMILES 
input:iqvia_data/drugbank_drugs_info.csv   
output:iqvia_data/drug2smiles.pkl   
```bash
### need optimize 
python src/drug2smiles.py 
```




### 3.4 Aggregation

input:     
* ctgov_data/diseases.csv  
* iqvia_data/drug2smiles.pkl  
* all_xml        

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


