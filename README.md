# Clinical Trial Outcome Prediction

















## 0. Download code repo

```bash 

git clone git@github.com:futianfan/clinical-trial-outcome-prediction.git

cd clinical-trial-outcome-prediction 

mkdir -p data save_model 

```























## 1. Installation via Conda 

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


### 3.1 Collect all the NCTIDs.
input: ClinicalTrialGov/   
output: data/all_xml 
```bash
find ClinicalTrialGov/ -name NCT*.xml | sort > data/all_xml
```
The current version has 348,891 trial IDs. 


### 3.2 diseaes -> icd10 using [ClinicalTable](https://clinicaltables.nlm.nih.gov/)
input: 
* ClinicalTrialGov/  
* data/all_xml   

output:	data/diseases.csv  
```bash 
python src/collect_disease_from_raw.py
```


### 3.3 drug -> SMILES using [DrugBank](https://go.drugbank.com/)


input: data/drugbank_drugs_info.csv   

output: data/drug2smiles.pkl   
```bash
python src/drug2smiles.py 
```



### 3.4 Aggregation

input:     
* data/diseases.csv  
* data/drug2smiles.pkl  
* data/all_xml         

output: data/raw_data.csv
```bash
python src/collect_raw_data.py | tee process.log 
```

























## 4. TOP: Trial Outcome Prediction benchmark Dataset 



### 4.1 Data Split 

input: data/raw_data.csv 


output: 
* data/phase_I_{train/valid/test}.csv 
* data/phase_II_{train/valid/test}.csv 
* data/phase_III_{train/valid/test}.csv 
* data/indication_{train/valid/test}.csv 


```bash
python src/data_split.py 
```


### 4.2 Data Statistics 

| Dataset  | \# Train | \# Valid | \# Test | \# Total | Split Date |
|-----------------|-------------|-------------|------------|-------------|------------|
| Phase I |  1028  |  146  |  295  |   1469  |  08/13/2014  | 
| Phase II | 2667 |  381 | 762  |   3810  |  03/20/2014  | 
| Phase III |  4286  |  612  |  1225 |  6123  |  04/07/2014  | 
| Indication |  3767  |  538  |  1077   |  5382  |  05/21/2014  | 


































## 5. Learn and Inference 





### 5.1 Phase I Prediction

```bash

python src/learn_phaseI.py

```


### 5.2 Phase II Prediction

```bash

python src/learn_phaseII.py


```

### 5.3 Phase III Prediction

```bash
python src/learn_phaseIII.py
```

### 5.4 Indication Prediction

```bash
python src/learn_indication.py 
```

### Result Table 

| Dataset  | PR-AUC | F1 | ROC-AUC |
|-----------------|-------------|-------------|------------|
| Phase I | 0.7495 (0.0277) | 0.8448 (0.0175) | 0.8146 (0.0233)   |    
| Phase II | 0.5585 (0.0253) | 0.5984 (0.0300) | 0.7619 (0.0214)  |    
| Phase III | 0.6199 (0.0165) | 0.6613 (0.0185) | 0.7171 (0.0134) |    
| Indication | 0.7037 (0.0136) | 0.7608 (0.0118) | 0.7849 (0.0114)  |   





 



















## Quick Start 

* 1. Installation via Conda
* 4. TOP: Trial Outcome Prediction benchmark Dataset 
* 5. Learn and Inference



## Contact

Please contact futianfan@gmail.com for help or submit an issue. This is a joint work with [Kexin Huang](https://www.kexinhuang.com/), [Cao(Danica) Xiao](https://sites.google.com/view/danicaxiao/), Lucas M. Glass and [Jimeng Sun](http://sunlab.org/). 

























