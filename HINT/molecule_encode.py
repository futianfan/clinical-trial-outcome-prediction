'''
input: 
	smiles batch
 


utility
	1. graph MPN
	2. smiles 
	3. morgan feature 

output:
	1. embedding batch 



deeppurpose
	DDI
	encoders  model 

to do 
	lst -> dataloader -> feature -> model 


	mpnn's feature -> collate -> model 

'''

import csv 
from tqdm import tqdm 
import numpy as np
from copy import deepcopy 
import matplotlib.pyplot as plt

import rdkit
import rdkit.Chem as Chem 
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.*')  
# from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import torch 
torch.manual_seed(0)
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data  #### data.Dataset 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from HINT.module import Highway 

def get_drugbank_smiles_lst():
	drugfile = 'data/drugbank_drugs_info.csv'
	with open(drugfile, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	return [row[27] for row in rows]

def txt_to_lst(text):
	"""
		"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
	"""
	text = text[1:-1]
	lst = [i.strip()[1:-1] for i in text.split(',')]
	return lst 

def get_cooked_data_smiles_lst():
	cooked_file = 'data/raw_data.csv'
	with open(cooked_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	smiles_lst = [row[8] for row in rows]
	smiles_lst = list(map(txt_to_lst, smiles_lst))
	from functools import reduce
	smiles_lst = list(reduce(lambda x,y:x+y, smiles_lst))
	smiles_lst = list(set(smiles_lst))
	# print(len(smiles_lst))  
	return smiles_lst



def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
### basic setting from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/mpn.py

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def smiles2mpnnfeature(smiles):
	## from mpn.py::tensorize  
	'''
		data-flow:   
			data_process(): apply(smiles2mpnnfeature)
			DBTA: train(): data.DataLoader(data_process_loader())
			mpnn_collate_func()
	'''
	padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
	fatoms, fbonds = [], [padding] 
	in_bonds,all_bonds = [], [(-1,-1)] 
	mol = get_mol(smiles)
	if mol is not None:
		n_atoms = mol.GetNumAtoms()
		for atom in mol.GetAtoms():
			fatoms.append( atom_features(atom))
			in_bonds.append([])

		for bond in mol.GetBonds():
			a1 = bond.GetBeginAtom()
			a2 = bond.GetEndAtom()
			x = a1.GetIdx() 
			y = a2.GetIdx()

			b = len(all_bonds)
			all_bonds.append((x,y))
			fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
			in_bonds[y].append(b)

			b = len(all_bonds)
			all_bonds.append((y,x))
			fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
			in_bonds[x].append(b)

		total_bonds = len(all_bonds)
		fatoms = torch.stack(fatoms, 0) 
		fbonds = torch.stack(fbonds, 0) 
		agraph = torch.zeros(n_atoms,MAX_NB).long()
		bgraph = torch.zeros(total_bonds,MAX_NB).long()
		for a in range(n_atoms):
			for i,b in enumerate(in_bonds[a]):
				agraph[a,i] = b

		for b1 in range(1, total_bonds):
			x,y = all_bonds[b1]
			for i,b2 in enumerate(in_bonds[x]):
				if all_bonds[b2][0] != y:
					bgraph[b1,i] = b2
	else: 
		# print('Molecules not found and change to zero vectors..')
		fatoms = torch.zeros(0,39)
		fbonds = torch.zeros(0,50)
		agraph = torch.zeros(0,6)
		bgraph = torch.zeros(0,6)
	Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
	shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
	return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor]


class smiles_dataset(data.Dataset):
	def __init__(self, smiles_lst, label_lst):
		self.smiles_lst = smiles_lst 
		self.label_lst = label_lst 

	def __len__(self):
		return len(self.smiles_lst)

	def __getitem__(self, index):
		smiles = self.smiles_lst[index]
		label = self.label_lst[index]
		smiles_feature = smiles2mpnnfeature(smiles)
		return smiles_feature, label 

## DTI.py --> collate 

## x is a list, len(x)=batch_size, x[i] is tuple, len(x[0])=5  
def mpnn_feature_collate_func(x): 
	return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]

def mpnn_collate_func(x):
	#print("len(x) is ", len(x)) ## batch_size 
	#print("len(x[0]) is ", len(x[0])) ## 3--- data_process_loader.__getitem__ 
	mpnn_feature = [i[0] for i in x]
	#print("len(mpnn_feature)", len(mpnn_feature), "len(mpnn_feature[0])", len(mpnn_feature[0]))
	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
	from torch.utils.data.dataloader import default_collate
	x_remain = [i[1:] for i in x]
	x_remain_collated = default_collate(x_remain)
	return [mpnn_feature] + x_remain_collated


def data_loader():
	smiles_lst = get_cooked_data_smiles_lst() 
	label_lst = [1 for i in range(len(smiles_lst))]	
	dataset = smiles_dataset(smiles_lst, label_lst)
	dataloader = data.DataLoader(dataset, batch_size=32, collate_fn = mpnn_collate_func, ) 
	return dataloader 


class MPNN(nn.Sequential):
	def __init__(self, mpnn_hidden_size, mpnn_depth, device):
		super(MPNN, self).__init__()
		self.mpnn_hidden_size = mpnn_hidden_size
		self.mpnn_depth = mpnn_depth 

		self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
		self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

		self.device = device
		self = self.to(self.device)

	def set_device(self, device):
		self.device = device 


	@property
	def embedding_size(self):
		return self.mpnn_hidden_size 

	### forward single molecule sequentially. 
	def feature_forward(self, feature):
		''' 
			batch_size == 1 
			feature: utils.smiles2mpnnfeature 
		'''
		fatoms, fbonds, agraph, bgraph, atoms_bonds = feature
		agraph = agraph.long()
		bgraph = bgraph.long()
		#print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape, atoms_bonds.shape)
		atoms_bonds = atoms_bonds.long()
		batch_size = atoms_bonds.shape[0]
		N_atoms, N_bonds = 0, 0 
		embeddings = []
		for i in range(batch_size):
			n_a = atoms_bonds[i,0].item()
			n_b = atoms_bonds[i,1].item()
			if (n_a == 0):
				embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
				embeddings.append(embed.to(self.device))
				continue 
			sub_fatoms = fatoms[N_atoms:N_atoms+n_a,:].to(self.device)
			sub_fbonds = fbonds[N_bonds:N_bonds+n_b,:].to(self.device)
			sub_agraph = agraph[N_atoms:N_atoms+n_a,:].to(self.device)
			sub_bgraph = bgraph[N_bonds:N_bonds+n_b,:].to(self.device)
			embed = self.single_feature_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
			embed = embed.to(self.device)           
			embeddings.append(embed)
			N_atoms += n_a
			N_bonds += n_b
		if len(embeddings)==0:
			return None 
		else:
			return torch.cat(embeddings, 0)

	def single_feature_forward(self, fatoms, fbonds, agraph, bgraph):
		'''
			fatoms: (x, 39)
			fbonds: (y, 50)
			agraph: (x, 6)
			bgraph: (y,6)
		'''
		### invalid molecule
		if fatoms.shape[0] == 0:
			return create_var(torch.zeros(1, self.mpnn_hidden_size).to(self.device))
		agraph = agraph.long()
		bgraph = bgraph.long()
		fatoms = create_var(fatoms).to(self.device)
		fbonds = create_var(fbonds).to(self.device)
		agraph = create_var(agraph).to(self.device)
		bgraph = create_var(bgraph).to(self.device)

		binput = self.W_i(fbonds)
		message = F.relu(binput)
		#print("shapes", fbonds.shape, binput.shape, message.shape)
		for i in range(self.mpnn_depth - 1):
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = F.relu(binput + nei_message)

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = F.relu(self.W_o(ainput))
		return torch.mean(atom_hiddens, 0).view(1,-1)


	def forward_single_smiles(self, smiles):
		fatoms, fbonds, agraph, bgraph, _ = smiles2mpnnfeature(smiles)
		embed = self.single_feature_forward(fatoms, fbonds, agraph, bgraph).view(1,-1)
		return embed 

	def forward_smiles_lst(self, smiles_lst):
		embed_lst = [self.forward_single_smiles(smiles) for smiles in smiles_lst]
		embed_all = torch.cat(embed_lst, 0)
		return embed_all

	def forward_smiles_lst_average(self, smiles_lst): 
		embed_all = self.forward_smiles_lst(smiles_lst)
		embed_avg = torch.mean(embed_all, 0).view(1,-1)
		return embed_avg


	def forward_smiles_lst_lst(self, smiles_lst_lst): 
		embed_lst = [self.forward_smiles_lst_average(smiles_lst) for smiles_lst in smiles_lst_lst]
		embed_all = torch.cat(embed_lst, 0)  #### n,dim
		return embed_all



class ADMET(nn.Sequential):

	def __init__(self, molecule_encoder, highway_num, device,  
					epoch, lr, weight_decay, save_name):
		super(ADMET, self).__init__()
		self.molecule_encoder = molecule_encoder 
		self.embedding_size = self.molecule_encoder.embedding_size
		self.highway_num = highway_num 
		self.highway_nn_lst = nn.ModuleList([Highway(size = self.embedding_size, num_layers = self.highway_num) for i in range(5)])
		self.fc_output_lst = nn.ModuleList([nn.Linear(self.embedding_size, 1) for i in range(5)])
		self.f = F.relu 
		self.loss = nn.BCEWithLogitsLoss()

		self.epoch = epoch 
		self.lr = lr 
		self.weight_decay = weight_decay 
		self.save_name = save_name 

		self.device = device 
		self = self.to(device)

	def set_device(self, device):
		self.device = device 
		self.molecule_encoder.set_device(device)


	def forward_smiles_lst_embedding(self, smiles_lst, idx):
		embed_all = self.molecule_encoder.forward_smiles_lst(smiles_lst)
		output = self.highway_nn_lst[idx](embed_all)
		return output 

	def forward_embedding_to_pred(self, embeded, idx):
		return self.fc_output_lst[idx](embeded)

	def forward_smiles_lst_pred(self, smiles_lst, idx):
		embeded = self.forward_smiles_lst_embedding(smiles_lst, idx)
		fc_output = self.forward_embedding_to_pred(embeded, idx)
		return fc_output   

	def test(self, dataloader_lst, return_loss = True):
		loss_lst = []
		for idx in range(1):
			single_loss_lst = []
			for smiles_lst, label_vec in dataloader_lst[idx]:
				output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
				loss = self.loss(output, label_vec.to(self.device).float())
				single_loss_lst.append(loss.item())
			loss_lst.append(np.mean(single_loss_lst))
		return np.mean(loss_lst)

	def train(self, train_loader_lst, valid_loader_lst):
		opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
		train_loss_record = [] 
		valid_loss = self.test(valid_loader_lst, return_loss=True)
		valid_loss_record = [valid_loss]
		best_valid_loss = valid_loss 
		best_model = deepcopy(self)
		for ep in tqdm(range(self.epoch)):
			data_iterator_lst = [iter(train_loader_lst[idx]) for idx in range(5)]
			try: 
				while True:
					for idx in range(1):
						smiles_lst, label_vec = next(data_iterator_lst[idx])
						output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
						loss = self.loss(output, label_vec.float()) 
						opt.zero_grad() 
						loss.backward()
						opt.step()	
			except:
				pass 
			valid_loss = self.test(valid_loader_lst, return_loss = True)
			valid_loss_record.append(valid_loss)						

			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss 
				best_model = deepcopy(self)

		self = deepcopy(best_model)





if __name__ == "__main__":
	model = MPNN(mpnn_hidden_size = 50, mpnn_depth = 3)
	dataloader = data_loader()
	for smiles_feature, labels in dataloader:
		embedding = model(smiles_feature) 
		print(embedding.shape)
		
	# smiles_lst = get_cooked_data_smiles_lst() 
	# valid_cnt, cnt = 0, 0 
	# for i,smiles in tqdm(enumerate(smiles_lst)):
	# 	feature = smiles2mpnnfeature(smiles)
	# 	if feature is not None:
	# 		valid_cnt += 1
	# 	if i%100==0:
	# 		print("valid rate is", str(valid_cnt/(i+1)))

	### single molecule forward 
	# for smiles in smiles_lst:
	# 	fatoms, fbonds, agraph, bgraph, abshape = smiles2mpnnfeature(smiles)
	# 	embedding = model.single_molecule_forward(fatoms, fbonds, agraph, bgraph)
	# 	print(embedding.shape)











