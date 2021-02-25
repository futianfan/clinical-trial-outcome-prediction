## 1. import 
## 2. input & hyperparameter
## 3. pretrain 
## 4. <5 line for 'dataloader, model build, train, inference'
################################################


## 1. import 
import torch, os 
from dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst, csv_three_feature_2_complete_dataloader
from molecule_encode import MPNN, ADMET 
from icdcode_encode import GRAM, build_icdcode2ancestor_dict
from protocol_encode import Protocol_Embedding
from model import Interaction, HINT_nograph, HINT, Only_Molecule,Only_Disease


## 2. input & hyperparameter
base_name_lst = ['trial', 'phase_I', 'phase_II', 'phase_III']

base_name = 'trial' 


train_file = 'ctgov_data/' + base_name + '_train.csv'
valid_file = 'ctgov_data/' + base_name + '_valid.csv'
test_file = 'ctgov_data/' + base_name + '_test.csv'
subgroup_lst = ['digest', 'nervous', 'infection', 'respiratory']
subgroup_file_lst = ['ctgov_data/' + base_name +'_' + subgroup + '_test.csv' for subgroup in subgroup_lst]


mpnn_model = MPNN(mpnn_hidden_size = 50, mpnn_depth=3)



# ## 3. pretrain 
# admet_model_path = "save_model/admet_model"
# if not os.path.exists(admet_model_path):
# 	admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
# 	admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
# 	admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
# 	admet_model = ADMET(mpnn_model = mpnn_model, 
# 						highway_num=2, 
# 						epoch=3, 
# 						lr=5e-4, 
# 						weight_decay=0, 
# 						save_name = 'admet_')
# 	admet_model.train(admet_trainloader_lst, admet_testloader_lst)
# 	torch.save(admet_model, admet_model_path)
# else:
# 	admet_model = torch.load(admet_model_path)
# exit()

## 4. <10 line for 'dataloader, model build, train, inference'
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 
test_complete_loader = csv_three_feature_2_complete_dataloader(test_file, shuffle=False, batch_size = 32)
subgroup_test_loader_lst = [csv_three_feature_2_dataloader(subgroup_file, shuffle=False, batch_size=32) for subgroup_file in subgroup_file_lst]
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict)
protocol_model = Protocol_Embedding(output_dim = 50, highway_num=3)

# model = HINT_nograph(molecule_encoder = mpnn_model, 
# 								 disease_encoder = gram_model, 
# 								 protocol_encoder = protocol_model,
# 								 global_embed_size = 50, 
# 								 highway_num_layer = 2,
# 								 prefix_name = base_name,  
# 								 epoch = 7,
# 								 lr = 5e-4, 
# 								 weight_decay = 0, 
# 								)

hint_model_path = "save_model/" + base_name + "_onlydisease.ckpt"
if not os.path.exists(hint_model_path):
	model = Only_Disease(molecule_encoder = mpnn_model, 
			 disease_encoder = gram_model, 
			 protocol_encoder = protocol_model,
			 global_embed_size = 50, 
			 highway_num_layer = 2,
			 prefix_name = base_name, 
			 epoch = 7,
			 lr = 1e-4, 
			 weight_decay = 0, 
			)
	model.learn(train_loader, valid_loader, test_loader)
	torch.save(model, hint_model_path)
else:
	model = torch.load(hint_model_path)
	model.test(test_loader, return_loss = False)

# # model.interpret(test_complete_loader)  #### interpret 
# for name,loader in zip(subgroup_lst, subgroup_test_loader_lst):
# 	try:
# 		auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio = model.test(loader, return_loss = False, validloader = valid_loader)
# 		print("=> " + name + "\nROC AUC => " + str(auc_score)[:4] + "\nF1 => " + str(f1score)[:4] + "\nPR-AUC => " + str(prauc_score)[:4]\
# 			 + "\nPrecision => " + str(precision)[:4] + "\nrecall => "+str(recall)[:4] + "\naccuracy => "+str(accuracy)[:4]\
# 			 + "\npredict 1 ratio => " + str(predict_1_ratio)[:4] + "\nlabel 1 ratio => " + str(label_1_ratio)[:4])
# 	except:
# 		pass 







