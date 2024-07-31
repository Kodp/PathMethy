import pandas as pd
import os
from tqdm import tqdm
import numpy as np

Kegg_path = "/mnt/mydisk/xjj/DNA_methylation/KEGG_cg/KEGG_result_DNA_methylation611.csv"
print("Kegg表格：")
Kegg_table = pd.read_csv(Kegg_path)
print(Kegg_table, '\n')

print('''Kegg_table['cg_count'].sum():''', Kegg_table['cg_count'].sum())

cgxxx_list = []

for cg_name in Kegg_table['cg_name']:  
  cgxxx_list.extend(cg_name.split("/"))
print(f"聚合cg列表的长度：{len(cgxxx_list)}, 不重复的cg的个数（Kegg所用的基因的个数）：{len(set(cgxxx_list))}")


base_path = '/mnt/mydisk/xjj/DNA_methylation/Result/7-subtype/GEO_data/'

new_path = '/mnt/mydisk/xjj/DNA_methylation/Result/7-subtype/GEO_data/'

file_list = [
'GSE_LGG.csv'
]


for file_name in file_list:
  print(f"read {base_path + file_name}")
  data = pd.read_csv(base_path + file_name, low_memory=False)
  KEGG_data_dict = {}
  
  
  for gene_idx, gene in tqdm(enumerate(cgxxx_list), desc=f"handle {file_name}", total=len(cgxxx_list)):

    KEGG_data_dict[f'cg{gene_idx+1}'] = data[gene]
    

  KEGG_data_dict['label'] = data['NewColumn']
  data_KEGG = pd.DataFrame(KEGG_data_dict)
  print(data_KEGG)
  file_name = 'Filtered' + file_name
  
  
  data_KEGG.to_csv(new_path + file_name, index=False)
