# Clinical Trial Outcome Prediction

















## 0. Download code repo

```bash 

git clone git@github.com:futianfan/clinical-trial-outcome-prediction.git

cd clinical-trial-outcome-prediction 

mkdir -p data save_model figure

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
It describes many important information about clinical trials, including NCT ID (i.e.,  identifiers to each clinical study), disease names, drugs, brief title and summary, phase, criteria, and statistical analysis results.  













































































## 3. Data Preprocess 


### 3.1 Collect all the NCTIDs.
input: ClinicalTrialGov/   
output: data/all_xml 
```bash
find ClinicalTrialGov/ -name NCT*.xml | sort > data/all_xml
```
The current version has 348,891 trial IDs. 


### 3.2 diseaes -> icd10 using 

* The diseases in ClinicalTrialGov are described in natural language. 

* On the other hand, [ICD-10](https://en.wikipedia.org/wiki/ICD-10) is the 10th revision of the International Statistical Classification of Diseases and Related Health Problems (ICD), a medical classification list by the World Health Organization (WHO). 
It leverages the hierarchical information inherent to medical ontologies. 

* We use [ClinicalTable](https://clinicaltables.nlm.nih.gov/), a public API to convert disease name (natural language) into ICD-10 code. 

input: 
* ClinicalTrialGov/  
* data/all_xml   

output:	data/diseases.csv  
```bash 
python src/collect_disease_from_raw.py
```


### 3.3 drug -> SMILES 

* The drugs in ClinicalTrialGov are described in natural language. 

* [DrugBank](https://go.drugbank.com/) contains rich information about drugs. 

* We use [DrugBank](https://go.drugbank.com/) to get the molecule structures of the drug. 

input: data/drugbank_drugs_info.csv   

output: data/drug2smiles.pkl   
```bash
python src/drug2smiles.py 
```



### 3.4 Aggregation

* input:     
** data/diseases.csv  
** data/drug2smiles.pkl  
** data/all_xml         

* output: 
** data/raw_data.csv
```bash
python src/collect_raw_data.py | tee data_process.log 
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

After processing the data, we learn the Hierarchical Interaction Network (HINT) on the following four tasks. The empirical results are given for reference. 



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

Mean and standard deviation are reported. 

| Dataset  | PR-AUC | F1 | ROC-AUC |
|-----------------|-------------|-------------|------------|
| Phase I | 0.7406 (0.0221) | 0.8474 (0.0144) |  0.8383 (0.0186) |    
| Phase II | 0.6030 (0.0198) | 0.7127 (0.0163) | 0.7850 (0.0136)  |    
| Phase III | 0.6279 (0.0165) | 0.6419 (0.0183) | 0.7257 (0.0109) |    
| Indication | 0.7136 (0.0120) | 0.7798 (0.0087) | 0.7987 (0.0111)  |   



### specify cpu or gpu in learn_X.py 
```python
device = torch.device("cuda:0")
device = torch.device("cpu")
```

 

 
 
















## Quick Reproduction

We provide a guidance for quick reproduction of the main experimental results. Since the data preprocessing procedure is time- and space-consuming, we make the processed data publicly available. 

* 0. Download code repo
* 1. Installation via Conda
* Download the [processed data](https://drive.google.com/drive/folders/1EJvVITNRdq4BYU6L27NF5f1moHqB8DTf?usp=sharing) from Google Drive. It requires 1.0 G. Please make sure that all the data files are in the folder "./data". 
* 5. Learn and Inference



## Contact

Please contact futianfan@gmail.com for help or submit an issue. This is a joint work with [Kexin Huang](https://www.kexinhuang.com/), [Cao(Danica) Xiao](https://sites.google.com/view/danicaxiao/), Lucas M. Glass and [Jimeng Sun](http://sunlab.org/). 

























