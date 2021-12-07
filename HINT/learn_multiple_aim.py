## 1. import 
## 2. input & hyperparameter
## 3. pretrain 
## 4. 'dataloader, model build, train, inference'
################################################


## 1. import 
import torch, os, sys
torch.manual_seed(0) 
sys.path.append('.')
from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst, csv_three_feature_2_complete_dataloader
from HINT.molecule_encode import MPNN, ADMET 
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel_multi
device = torch.device("cpu")
if not os.path.exists("figure"):
	os.makedirs("figure")




## 2. data
base_name = 'phase_II' 
base_name = "indication"
# base_name = 'phase_III'
# base_name = "toy" 
datafolder = "auxiliary_data"
train_file = os.path.join(datafolder, base_name + '_train.csv')
valid_file = os.path.join(datafolder, base_name + '_valid.csv')
test_file = os.path.join(datafolder, base_name + '_test.csv')





## 3. pretrain 
mpnn_model = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)
admet_model_path = "save_model/admet_model.ckpt"
if not os.path.exists(admet_model_path):
	admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
	admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
	admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
	admet_model = ADMET(molecule_encoder = mpnn_model, 
						highway_num=2, 
						device = device, 
						epoch=3, 
						lr=5e-4, 
						weight_decay=0, 
						save_name = 'admet_')
	admet_model.train(admet_trainloader_lst, admet_testloader_lst)
	torch.save(admet_model, admet_model_path)
else:
	admet_model = torch.load(admet_model_path)
	admet_model = admet_model.to(device)
	admet_model.set_device(device)




## 4. dataloader, model build, train, inference
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 




icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
protocol_model = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)



hint_model_path = "save_model2/" + base_name + ".ckpt"
if not os.path.exists(hint_model_path):
	model = HINTModel_multi(molecule_encoder = mpnn_model, 
			 disease_encoder = gram_model, 
			 protocol_encoder = protocol_model,
			 device = device, 
			 global_embed_size = 50, 
			 highway_num_layer = 2,
			 prefix_name = base_name, 
			 # gnn_hidden_size = 50,  
			 epoch = 5,
			 lr = 1e-3, 
			 weight_decay = 0, 
			)
	# model.init_pretrain(admet_model)
	pred_all, label_all = model.learn(train_loader, valid_loader, test_loader)
	# model.bootstrap_test(test_loader)
	# torch.save(model, hint_model_path)
else:
	model = torch.load(hint_model_path)
	model.bootstrap_test(test_loader)




import matplotlib.pyplot as plt
import numpy as np 
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
lst = np.array([0,1,2,3])
cm = confusion_matrix(label_all, pred_all, labels=lst)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lst)
disp.plot()
plt.xlabel("predicted label", fontsize = 21)
plt.ylabel("True label", fontsize = 20)
plt.tight_layout()
plt.savefig("figure/"+base_name + "_cm.png")











