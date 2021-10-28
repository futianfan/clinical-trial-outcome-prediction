# Benchmark

To standardize the clinical trial approval prediction, we create a benchmark dataset for Trial Approval Prediction named TAP, which incorporate rich data components about clinical trials, including drug, disease and protocol (eligibility criteria). 
Benchmark can be mainly divided into two parts:
- `Raw Data` describes all the data sources. 
  - `ClinicalTrial.gov`: all the clinical trials records. 
  - `DrugBank`: molecule structures of all the drugs. 
  - `ClinicalTable`: API for ICD-10 codes. 
  - `MoleculeNet`: ADMET data. 
- `Data Curation Process` describes data curation process.
  - Collect all the records
  - diseases to icd10 
  - drug to SMILES 
  - ICD-10 code hierarchy
  - Sentence Embedding for trial protocol 
  - Selection of clinical trial
  - Data split 
  - Statistics of Dataset 


## Raw Data 

### ClinicalTrial.gov
- description
  - We download all the clinical trials records from [ClinicalTrial.gov](https://clinicaltrials.gov/AllPublicXML.zip). 
It contains 348,891 clinical trial records. The data size grows with time because more clinical trial records are added. 
It describes many important information about clinical trials, including NCT ID (i.e.,  identifiers to each clinical study), disease names, drugs, brief title and summary, phase, criteria, and statistical analysis results. 

- output
  - `./raw_data`: store all the xml files for all the trials (identified by NCT ID).  
  - **TrialTrove**: `./trialtrove/trial_outcomes_v1.csv`. We do not release the real trial approval label due to privacy issue. When the real label is not available, we use another method to get rough label via leveraging the statistical test in clinicaltrials.gov. When the `p-value` is smaller than 0.05, we take it as positive sample. Please see `benchmark/pseudolabel.py`. 


```bash 
mkdir -p raw_data
cd raw_data
wget https://clinicaltrials.gov/AllPublicXML.zip
```


Then we unzip the ZIP file. The unzipped file occupies over 8.6 G. Please make sure you have enough space. 
```bash 
unzip AllPublicXML.zip
cd ../
```

### DrugBank

- description
  - We use [DrugBank](https://go.drugbank.com/) to get the molecule structures ([SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), simplified molecular-input line-entry system) of the drug. 

- input
  - None 

- output
  - `data/drugbank_drugs_info.csv `  

### ClinicalTable

[ClinicalTable](https://clinicaltables.nlm.nih.gov/) is a public API to convert disease name (natural language) into ICD-10 code. 

### MoleculeNet
- description
  - [MoleculeNet](https://moleculenet.org/) include five datasets across the main categories of drug pharmaco-kinetics (PK). For absorption, we use the bioavailability dataset. For distribution, we use the blood-brain-barrier experimental results provided. For metabolism, we use the CYP2C19 experiment paper, which is hosted in the PubChem biassay portal under AID 1851. For excretion, we use the clearance dataset from the eDrug3D database. For toxicity, we use the ToxCast dataset, provided by MoleculeNet. We consider drugs that are not toxic across all toxicology assays as not toxic and otherwise toxic. 

- input
  - None 

- output 
  - `data/ADMET`

---

## Data Curation Process 

### Collect all the records
- description
  - download all the records from clinicaltrial.gov. The current version has 348,891 trial IDs. 

- input
  - `raw_data/`: raw data, store all the xml files for all the trials (identified by NCT ID).   

- output
  - `data/all_xml`: store NCT IDs for all the xml files for all the trials.  

```bash
find raw_data/ -name NCT*.xml | sort > data/all_xml
```


### Disease to ICD-10 code

- description

  - The diseases in [ClinicalTrialGov](clinicaltrials.gov) are described in natural language. 

  - On the other hand, [ICD-10](https://en.wikipedia.org/wiki/ICD-10) is the 10th revision of the International Statistical Classification of Diseases and Related Health Problems (ICD), a medical classification list by the World Health Organization (WHO). It leverages the hierarchical information inherent to medical ontologies. 

  - We use [ClinicalTable](https://clinicaltables.nlm.nih.gov/), a public API to convert disease name (natural language) into ICD-10 code. 

- input 
  - `raw_data/ ` 
  - `data/all_xml`   

- output
  -	`data/diseases.csv ` 

```bash 
python benchmark/collect_disease_from_raw.py
```


### drug to SMILES 

- description
  - [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) is simplified molecular-input line-entry system of the molecule. 

  - The drugs in [ClinicalTrialGov](clinicaltrials.gov) are described in natural language. 

  - [DrugBank](https://go.drugbank.com/) contains rich information about drugs. 

  - We use [DrugBank](https://go.drugbank.com/) to get the molecule structures in terms of SMILES. 

- input
  - `data/drugbank_drugs_info.csv `  

- output
  - `data/drug2smiles.pkl `  

```bash
python benchmark/drug2smiles.py 
```



### Selection of clinical trial

We design the following inclusion/exclusion criteria to select eligible clinical trials for learning. 

- inclusion criteria 
  - study-type is interventional 
  - intervention-type is drug
  - p-value in primary-outcome is available
  - disease codes are available 
  - drug molecules are available 
  - eligibility criteria are available


- exclusion criteria 
  - study-type is observational 
  - intervention-type is surgery, biological, device
  - p-value in primary-outcome is not available
  - disease codes are not available 
  - drug molecules are not available 
  - eligibility criteria are not available

The csv file contains following features:

* `nctid`: NCT ID, e.g., NCT00000378, NCT04439305. 
* `status`: `completed`, `terminated`, `active, not recruiting`, `withdrawn`, `unknown status`, `suspended`, `recruiting`. 
* `why_stop`: for completed, it is empty. Otherwise, the common reasons contain `slow/low/poor accrual`, `lack of efficacy`
* `label`: 0 (failure) or 1 (approved).  
* `phase`: I, II, III or IV. 
* `diseases`: list of diseases. 
* `icdcodes`: list of icd-10 codes.
* `drugs`: list of drug names
* `smiless`: list of SMILES
* `criteria`: egibility criteria 

- input    
  - `data/diseases.csv ` 
  - `data/drug2smiles.pkl`  
  - `data/all_xml ` 
  - `trialtrove/*`       

- output 
  - `data/raw_data.csv`


```bash
python benchmark/collect_raw_data.py | tee data_process.log 
```





<p align="center"><img src="./figure/dataset.png" alt="logo" width="650px" /></p>




### Data Split 
- description (Split criteria)
  - phase I: phase I trials, augmented with phase IV trials as positive samples. 
  - phase II: phase II trials, augmented with phase IV trials as positive samples.  
  - phase III: phase III trials, augmented with failed phase I and II trials as negative samples and successed phase IV trials as positive samples. 
  - indication: trials that fail in phase I or II or III are negative samples. Trials that pass phase III or enter phase IV are positive samples.  
- input
  - `data/raw_data.csv` 

- output: 
  - `data/phase_I_{train/valid/test}.csv` 
  - `data/phase_II_{train/valid/test}.csv` 
  - `data/phase_III_{train/valid/test}.csv` 
  - `data/indication_{train/valid/test}.csv` 


```bash
python benchmark/data_split.py 
```


### ICD-10 code hierarchy 

- description 
  - get all the ancestor code for the current icd-10 code. 

- input
  - `data/raw_data.csv` 

- output: 
  - `data/icdcode2ancestor_dict.pkl` 


```bash 
python benchmark/icdcode_encode.py 
```

### Sentence embedding 

- description 
  - BERT embedding to get sentence embedding for sentence in clinical protocol. 

- input
  - `data/raw_data.csv` 

- output: 
  - `data/sentence2embedding.pkl` 


```bash 
python benchmark/protocol_encode.py 
```


### Statistics of Dataset

| Settings  | Train Pass | Train Failure | Test Pass | Test Failure |  Split date |  
|-----------------|-------------|-------------|------------|-------------|------------|
| Phase I  | 1,920 | 702 | 534 | 217  |  Aug 13, 2014 | 
| Phase II  | 3,540 | 2,856 | 1,151 | 678 |  March 20, 2014 | 
| Phase III  | 3,445 | 3,891 | 1,184 | 913 | April 7, 2014 | 
| Indication | 3,561 | 3,257 | 1,083 | 865 | May 21, 2014 | 


We use temporal split, where the earlier trials (before split date) are used for training and validation, the later trials (after split date) are used for testing. 




## Contact

Please contact futianfan@gmail.com for help or submit an issue. This is a joint work with [Kexin Huang](https://www.kexinhuang.com/), [Cao(Danica) Xiao](https://sites.google.com/view/danicaxiao/), Lucas M. Glass and [Jimeng Sun](http://sunlab.org/). 


















