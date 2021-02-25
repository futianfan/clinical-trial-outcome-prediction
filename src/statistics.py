import csv 
from dataloader import smiles_txt_to_lst,icdcode_text_2_lst_of_lst
file = "ctgov_data/raw_data.csv"
###  nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria 
with open(file, 'r') as csvfile:
	rows = list(csv.reader(csvfile, delimiter=','))[1:]
disease_cnt, icdcode_cnt, drug_cnt = 0,0,0

icdcode_lst = [icdcode_text_2_lst_of_lst(i[6]) for i in rows]
print(icdcode_lst[:4])





