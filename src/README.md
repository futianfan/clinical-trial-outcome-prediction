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


<details>
  <summary>Click here for the code!</summary>

```python
from DeepPurpose import PPI as models
```

</details>

- model architecture 
  - `model.py`
    - three model classes, build model from simple to complex. 



<details>
  <summary>Click here for the code!</summary>

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

</details>



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
    - mpn features 
    ```python
    def smiles2mpnnfeature(smiles):
    	... 
    ```
    - dataloader
    ```python
    from torch.utils import data   
    class smiles_dataset(data.Dataset):
		...
	def mpnn_collate_func(x):
		...
    ```
    - mpn 
    ```python
    from torch import nn
    class MPNN(nn.Sequential):
    	...
    ```
  - `protocol_encode.py`
  	- preprocess 
  	```python
  	def save_sentence_bert_dict_pkl():
		cleaned_sentence_set = collect_cleaned_sentence_set() 
		from biobert_embedding.embedding import BiobertEmbedding
		biobert = BiobertEmbedding()
		def text2vec(text):
			return biobert.sentence_vector(text)
		protocol_sentence_2_embedding = dict()
		for sentence in tqdm(cleaned_sentence_set):
			protocol_sentence_2_embedding[sentence] = text2vec(sentence)
		pickle.dump(protocol_sentence_2_embedding, open('data/sentence2embedding.pkl', 'wb'))
		return 

	if __name__ == "__main__":
		save_sentence_bert_dict_pkl() 
  	```
  	- protocol embeddor
  	```python
  	from torch import nn 
  	class Protocol_Embedding(nn.Sequential):
  		...
  	```
  - `gnn_layers.py` contains standard implementation of existing GNN's building block (**single layer gnn**).
    - Graph Convolutional Network 
    ```python
    from torch.nn.modules.module import Module
    class GraphConvolution(Module):
    	...
    ```
    - Graph Attention Network
    ```python
    from torch.nn.modules.module import Module
	class GraphAttention(nn.Module):
		...
    ```
  - `module.py` contains standard implementation of existing neural module, e.g., highway, GCN
  	- Highway Network 
  	```python
  	import torch.nn as nn
  	class Highway(nn.Module):
  		def __init__(self, ...):
  			...
  		def forward(self, ...):
  			...
  	```
  	- GCN 
  	```python
  	import torch.nn as nn
  	class GCN(nn.Module):
  		def __init__(self, ...):
  			...
  		def forward(self, ...):
  			...  	
  	```




