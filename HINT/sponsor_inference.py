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
from HINT.model import Interaction, HINT_nograph, HINTModel
device = torch.device("cpu")
if not os.path.exists("figure"):
	os.makedirs("figure")




## 2. data
base_name = 'phase_III' 

for base_name in ['phase_III', 'phase_II', 'phase_I']: 
	datafolder = "data"
	train_file = os.path.join(datafolder, base_name + '_train.csv')
	valid_file = os.path.join(datafolder, base_name + '_valid.csv')
	test_file = os.path.join(datafolder, base_name + '_test.csv')
	ongoing_file = "data/ongoing_"+base_name+".csv" 


	## 4. dataloader, model build, train, inference
	train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
	valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
	test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 
	ongoing_loader = csv_three_feature_2_dataloader(ongoing_file, shuffle=False, batch_size=32) 

	icdcode2ancestor_dict = build_icdcode2ancestor_dict()
	gram_model = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
	protocol_model = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)


	hint_model_path = "save_model/" + base_name + ".ckpt"
	# if not os.path.exists(hint_model_path):
	if True:
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




		model = HINTModel(molecule_encoder = mpnn_model, 
				 disease_encoder = gram_model, 
				 protocol_encoder = protocol_model,
				 device = device, 
				 global_embed_size = 50, 
				 highway_num_layer = 1,
				 prefix_name = base_name, 
				 gnn_hidden_size = 50,  
				 epoch = 2,
				 lr = 1e-3, 
				 weight_decay = 0, 
				)
		model.init_pretrain(admet_model)
		model.learn(train_loader, valid_loader, test_loader)



	nctid_lst, predict_lst = model.ongoing_test(test_loader)
	with open('data/test_predict_' + base_name + '.txt', 'w') as fout:
		for nctid, predict in zip(nctid_lst, predict_lst):
			fout.write(nctid+'\t'+str(predict)+'\n')


	nctid_lst, predict_lst = model.ongoing_test(ongoing_loader)
	with open('data/ongoing_predict_' + base_name + '.txt', 'w') as fout:
		for nctid, predict in zip(nctid_lst, predict_lst):
			fout.write(nctid+'\t'+str(predict)+'\n')













