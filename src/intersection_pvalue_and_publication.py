import os 

## input 
pvalue_file = "ctgov_data/nctid_with_pvalue"
pub_file = "nctid_publication_abstract/nctid2puburl.txt"
pub_folder = 'nctid_publication_abstract'

## output
pvalue_and_pub_file = "ctgov_data/nctid_with_pvalue_and_pub.txt"


with open(pvalue_file, 'r') as fin:
	lines = fin.readlines()
	nctid_with_pvalue_lst = [line.split('.')[0] for line in lines]

with open(pub_file, 'r') as fin:
	lines = fin.readlines() 
	nctid_with_pub_lst = [line.split()[0] for line in lines]

nctid_with_pvalue_set = set(nctid_with_pvalue_lst) 
nctid_with_pub_set = set(nctid_with_pub_lst)
nctid_with_pvalue_and_pub_set = nctid_with_pvalue_set.intersection(nctid_with_pub_set)


def filter_good_pub(nctid):
	file = os.path.join(pub_folder, nctid + '.txt')
	if not os.path.exists(file):
		return False 
	with open(file, 'r') as fin:
		lines = fin.readlines()
	if len(lines) >= 6:
		return True 
	return False

nctid_with_pvalue_and_good_pub_set = list(filter(filter_good_pub, list(nctid_with_pvalue_and_pub_set)))




with open(pvalue_and_pub_file, 'w') as fout:
	# for nctid in nctid_with_pvalue_and_pub_set:
	for nctid in nctid_with_pvalue_and_good_pub_set:
		fout.write(nctid + '\n')
print("number of nctid with both pvalue and good publication is ", len(nctid_with_pvalue_and_good_pub_set))




### calculate the ratio of good publication. 
with open(pub_file, 'r') as fin:
	lines = fin.readlines() 
	nctid_with_pub_lst = [line.split()[0] for line in lines]

cnt, good_cnt = 0, 0
for nctid in nctid_with_pub_lst:
	file = os.path.join(pub_folder, nctid + '.txt')
	if not os.path.exists(file):
		continue 
	cnt += 1
	with open(file, 'r') as fin:
		lines = fin.readlines()	
	if len(lines) >=6:
		good_cnt += 1

print("good pub: ", str(good_cnt), "total pub:", str(cnt), "good ratio is", str(1.0*good_cnt/cnt)[:5])








