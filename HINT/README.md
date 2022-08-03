# HINT: Learn and Inference 


After processing the data, we learn the Hierarchical Interaction Network (HINT) on the following four tasks. The following figure illustrates the pipeline of HINT. All the scripts are available in the folder `HINT`. 


<p align="center"><img src="./hint.png" alt="logo" width="810px" /></p>


### Tutorial (Jupyter Notebook) 

[`tutorial_HINT.ipynb`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/tutorial_HINT.ipynb) is a tutorial to learn and evaluate HINT step by step. 



### Phase I/II/III Prediction (Bash command line)

Phase-level prediction predicts the success probability of a single phase study. 

```bash
python HINT/learn_phaseI.py
```


```bash
python HINT/learn_phaseII.py
```


```bash
python HINT/learn_phaseIII.py
```






### METRICS

- **PR-AUC** (Precision-Recall Area Under Curve). Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
- **F1**. The F1 score is the harmonic mean of the precision and recall.
- **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve). ROC curve summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds. 


<!-- ### Result 

The empirical results are given for reference. The mean and standard deviation of 5 independent runs are reported. 

| Dataset  | PR-AUC | F1 | ROC-AUC |
|-----------------|-------------|-------------|------------|
| Phase I | 0.745 (0.009) | 0.820 (0.007) |  0.726 (0.009) |    
| Phase II | 0.685 (0.011) | 0.754 (0.010) | 0.698 (0.008)  |    
| Phase III | 0.709 (0.009) | 0.757 (0.008) | 0.784 (0.009) |    --> 



## Contact

Please contact futianfan@gmail.com for help or submit an issue. This is a joint work with [Kexin Huang](https://www.kexinhuang.com/), [Cao(Danica) Xiao](https://sites.google.com/view/danicaxiao/), Lucas M. Glass and [Jimeng Sun](http://sunlab.org/). 


## Code Architecture


- learn and inference on various task
  - `learn_phaseI.py`: predict whether the trial can pass phase I. 
  - `learn_phaseII.py`: predict whether the trial can pass phase II.
  - `learn_phaseIII.py`: predict whether the trial can pass phase III.
  - `learn_indication.py`: predict whether the trial can pass the indication (phase I-III).
- model architecture 
  - `model.py`
    - three model classes (`Interaction`, `HINT_nograph`, `HINTModel`), build model from simple to complex. 
  - `icdcode_encode.py` 
    - preprocess ICD-10 code, building ontology of icd-10 codes.
    - GRAM to model hierarchy of icd-10 code. 
  - `molecule_encode.py`
    - message passing network (MPN)
  - `protocol_encode.py`
    - protocol embeddor 
  - `module.py` contains standard implementation of existing neural module, e.g., highway, GCN
    - Highway Network 
    - Graph Convolutional Network (GCN) 



















