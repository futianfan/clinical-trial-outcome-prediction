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
    - three model classes 
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
  - `molecule_encode.py`
  - `protocol_encode.py`
  - `gnn_layers.py`
  - `module.py` 
