import networkx as nx
import matplotlib.pyplot as plt
from random import uniform  
import numpy as np 
import torch 

'''
nodes = ["molecule", "disease", "criteria", 'A', 'D', 'M', 'E', 'T', 'ADMET', 'DR', 'INTERACTION', 'CIM', "final"]
edges = [('A', 'ADMET'), ('D', 'ADMET'), ('M', 'ADMET'), ('E', 'ADMET'), ('T', 'ADMET'), 
			('DR', 'CIM'), ('INTERACTION', 'CIM'), ('CIM', 'final'), ('ADMET', 'final'), 
			("molecule", "A"), ("molecule", "D"), ("molecule", "M"), ("molecule", "E"), ("molecule", "T"),
			("molecule", "INTERACTION"), ("disease", "INTERACTION"), ("criteria", "INTERACTION"), 
			("disease", "DR"),  
			("disease", "molecule"), ("disease", "criteria"), ("molecule", "criteria")]
nodes2pos = {}
nodes2pos["molecule"] = [0, 0]
nodes2pos['disease'] = [0.5, -0.3]
nodes2pos['criteria'] = [1, 0]
nodes2pos['A'] = [-0.3, 1]
nodes2pos['D'] = [-0.1, 1]
nodes2pos['M'] = [0.1, 1]
nodes2pos['E'] = [0.3, 1]
nodes2pos['T'] = [0.5, 1]
nodes2pos['DR'] = [0.8, 1]
nodes2pos['INTERACTION'] = [1.1, 1]
nodes2pos['ADMET'] = [0, 2]
nodes2pos['CIM'] = [1, 2]
nodes2pos['final'] = [0.5, 3]
'''

def single_value_dictionary(dictionary):
	assert len(dictionary)==1
	return list(dictionary.values())[0]


def data2graph(attention_matrix, adj, save_name, text = None):
	'''
		attention_matrix: N*N 
		adj: N*N 
		save_name "xxx.png"
	'''
	plt.clf()
	plt.axis([-0.5, 1.2, -0.7, 3.4])

	# lst = ["molecule", "disease", "criteria", 'INTERACTION', 'risk_disease', 'augment_interaction', 'A', 'D', 'M', 'E', 'T', 'PK', "final"]

	nodes = ["molecule", "disease", "protocol", 'Inter', 'DR', 'Aug-Inter', 'A', 'D', 'M', 'E', 'T', 'PK',  "final"]
	colors = ['green', 'green', 'green', 'blue', 'blue', 'yellow', 'blue', 'blue', 'blue', 'blue', 'blue', 'yellow', 'grey']
	edges = [('A', 'PK'), ('D', 'PK'), ('M', 'PK'), ('E', 'PK'), ('T', 'PK'), 
				('DR', 'Aug-Inter'), ('Inter', 'Aug-Inter'), ('Aug-Inter', 'final'), ('PK', 'final'), 
				("molecule", "A"), ("molecule", "D"), ("molecule", "M"), ("molecule", "E"), ("molecule", "T"),
				("molecule", "Inter"), ("disease", "Inter"), ("protocol", "Inter"), 
				("disease", "DR"),  
				("disease", "molecule"), ("disease", "protocol"), ("molecule", "protocol")]
	#### 2d coordinate in graph 
	nodes2pos = {}
	nodes2pos["molecule"] = [0, 0]
	nodes2pos['disease'] = [0.5, -0.3]
	nodes2pos['protocol'] = [1, 0]
	nodes2pos['A'] = [-0.3, 1]
	nodes2pos['D'] = [-0.1, 1]
	nodes2pos['M'] = [0.1, 1]
	nodes2pos['E'] = [0.3, 1]
	nodes2pos['T'] = [0.5, 1]
	nodes2pos['DR'] = [0.8, 1]
	nodes2pos['Inter'] = [1.1, 1]
	nodes2pos['PK'] = [0, 2]
	nodes2pos['Aug-Inter'] = [1, 2]
	nodes2pos['final'] = [0.5, 3]
	g = nx.Graph()
	g.clear() 
	for idx, node in enumerate(nodes):
		g.add_node(node)
	node_size = attention_matrix.shape[0]
	for i in range(node_size):
		for j in range(i): 
			if adj[i,j].item()==1:
				g.add_edge(nodes[i],nodes[j],w=str(attention_matrix[i,j].item())[:4])
	### node position 
	pos = nx.spring_layout(g)
	for node in nodes:
		pos[node] = np.array(nodes2pos[node])
	nx.draw(g, pos, node_color=colors, node_size = 1500)
	# generate node_labels manually
	'''node_labels = {}
	for node in g.nodes:
		node_labels[node] = single_value_dictionary(g.nodes[node]) # G.nodes[node] will return all attributes of node 
	nx.draw_networkx_labels(g, pos, labels=node_labels)'''
	nx.draw_networkx_labels(g, pos)	
	edge_labels = {}
	for edge in g.edges:
		edge_labels[edge] = single_value_dictionary(g[edge[0]][edge[1]]) # G[edge[0]][edge[1]] will return all attributes of edge
	nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, )


	#nx.draw_networkx_edge_labels(g, pos)
	#plt.title(text) 
	#plt.text(-0.49, 3.2, text, fontsize=18)
	plt.savefig(save_name)
	return 



if __name__ == "__main__":
	##### original graph
	lst = ['A', 'D', 'M', 'E', 'T', 'ADMET', 'DR', 'INTERACTION', 'CIM', "final"]
	edge_lst = [('A', 'ADMET'), ('D', 'ADMET'), ('M', 'ADMET'), ('E', 'ADMET'), ('T', 'ADMET'), 
				('DR', 'CIM'), ('INTERACTION', 'CIM'), ('CIM', 'final'), ('ADMET', 'final')]
	#### after adding 3 feature node 
	lst = ["molecule", "disease", "criteria"] + lst
	edge_lst.extend([("molecule", "A"), ("molecule", "D"), ("molecule", "M"), ("molecule", "E"), ("molecule", "T"),
					 ("molecule", "INTERACTION"), ("disease", "INTERACTION"), ("criteria", "INTERACTION"), 
					 ("disease", "DR"),  
					 ("disease", "molecule"), ("disease", "criteria"), ("molecule", "criteria"), ])
	adj = torch.zeros(len(lst), len(lst))
	adj = torch.eye(len(lst)) * len(lst)
	num2str = {k:v for k,v in enumerate(lst)}
	str2num = {v:k for k,v in enumerate(lst)}
	for i,j in edge_lst:
		n1,n2 = str2num[i], str2num[j]
		adj[n1,n2] = 1
		adj[n2,n1] = 1
	node_size = len(lst)
	attention_matrix = torch.rand(node_size, node_size)
	save_name = "try.png"
	data2graph(attention_matrix, adj, save_name, text = "bilibili "*20)










