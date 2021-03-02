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
			...

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
<details>
  <summary><font color=red>Click here for the code!</font></summary>

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

</details>

    - GRAM to model hierarchy of icd-10 code. 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch import nn 
class GRAM(nn.Sequential):
	def __init__(self, ...):
		...

	def forward(self, ...):
		...
```

</details>

  - `molecule_encode.py`
    - mpn features 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
def smiles2mpnnfeature(smiles):
	... 
```

</details>

    - dataloader
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch.utils import data   
class smiles_dataset(data.Dataset):
	...
def mpnn_collate_func(x):
	...
```

</details>

    - mpn 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch import nn
class MPNN(nn.Sequential):
	...
```

</details>

  - `protocol_encode.py`
    - preprocess 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

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

</details>

    - protocol embeddor
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch import nn 
class Protocol_Embedding(nn.Sequential):
	...
```

</details>

  - `gnn_layers.py` contains standard implementation of existing GNN's building block (**single layer gnn**).
    - Graph Convolutional Network 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch.nn.modules.module import Module
class GraphConvolution(Module):
	...
```
</details>

    - Graph Attention Network
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
from torch.nn.modules.module import Module
class GraphAttention(nn.Module):
	...
```

</details>

  - `module.py` contains standard implementation of existing neural module, e.g., highway, GCN
    - Highway Network 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
import torch.nn as nn
class Highway(nn.Module):
	def __init__(self, ...):
		...
	def forward(self, ...):
		...
  	```

</details>

    - GCN 
<details>
  <summary><font color=red>Click here for the code!</font></summary>

```python
import torch.nn as nn
class GCN(nn.Module):
	def __init__(self, ...):
		...
	def forward(self, ...):
		...  	
```

</details>




