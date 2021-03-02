# Code for Hierarchical Interaction Graph 



## data preprocessing 

- data script

  - `collect_all.py`
  - `data_split.py`



## model 

- learn and inference on various task
  - `learn_phaseI.py`: predict whether the trial can pass phase I. 
  - `learn_phaseII.py`: predict whether the trial can pass phase II.
  - `learn_phaseIII.py`: predict whether the trial can pass phase III.
  - `learn_indication.py`: predict whether the trial can pass the indication (phase I-III).

- model architecture 
  - `model.py`
    - three model classes, build model from simple to complex. 
    ```python
    from torch import nn 
    class Interaction(nn.Sequential):

    	def __init__(self, ...):
			super(Interaction, self).__init__()
			... 

		def forward(self, ...):

    	def evaluation(self, ...):
    		...

    	def bootstrap_test(self, ...):
    		... 

    class HINT_nograph(Interaction):
    	def __init__(self, ...):
			super(HINT_nograph, self).__init__(....,) 
			...

		def forward(self, ...):
			...
	class HINT(HINT_nograph):
    	def __init__(self, ...):
			super(HINT, self).__init__(....,) 
			...

		def forward(self, ):
			... 
    ```
  - `icdcode_encode.py` 
    - preprocess ICD-10 code, building ontology of icd-10 codes.
    ```python
def build_icdcode2ancestor_dict():
	pkl_file = "data/icdcode2ancestor_dict.pkl"
	if os.path.exists(pkl_file):
		icdcode2ancestor = pickle.load(open(pkl_file, 'rb'))
		return icdcode2ancestor 
	all_code = collect_all_icdcodes() 
	icdcode2ancestor = defaultdict(list)
	for code in all_code:
		find_ancestor_for_icdcode(code, icdcode2ancestor)
	pickle.dump(icdcode2ancestor, open(pkl_file,'wb'))
	return icdcode2ancestor 

if __name__ == '__main__':
	dic = build_icdcode2ancestor_dict()     
    ```
    - GRAM model to model hierarchy of icd-10 code. 
    ```python
from torch import nn 
class GRAM(nn.Sequential):
    ```
  - `molecule_encode.py`
  - `protocol_encode.py`
  - `gnn_layers.py`
  - `module.py` 
