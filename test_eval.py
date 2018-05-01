# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:47:50 2018

@author: Natalia
"""
import pandas as pd
import pickle
import ast

folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
#csv_file = folder+'training/aligned/corpus/string_matching_output/AlignedCauses_2006-2012full_string_matched.csv'
csv_file = folder+'training/aligned/corpus/string_matching_output/test_ordered.csv'
df_system = pd.read_csv(csv_file, sep=";")


df_system = df_system.head(5000)
docIds = df_system["DocID"].tolist()


lan = "FR"
folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
## aligned
textf = folder+'training/aligned/corpus/AlignedCauses_2006-2012full.csv'
df = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 

df_gold=df[df["DocID"].isin(docIds)]
df_gold.to_csv("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/FR_dic_search_aligned_gold_5K.csv", sep=";", index=False)



df_output = pd.DataFrame(columns=['DocID', 'YearCoded', 'Gender', 'Age', 'LocationOfDeath', 'LineID', 'StandardText', 'IntType', 'IntValue', 'FoundEvidence', 'FoundEvidence2', 'ExtractedCode'])

index_output = 0
for index, row in df_system.iterrows():
    icd_codes = ast.literal_eval(row['dictionary_lookup'])
    icd_texts = row['full_dictionary_lookup']
    for code in icd_codes:
        df_output.loc[index_output] = [row['DocID'], row['YearCoded'], row['Gender'], row['Age'], row['LocationOfDeath'], row['LineID'], row['StandardText'], row['IntType'], row['IntValue'],"","",code]
        index_output=index_output+1
        
df_output.DocID = df_output.DocID.astype(int)
df_output.LineID = df_output.LineID.astype(int)
df_output.YearCoded = df_output.YearCoded.astype(int)
df_output.to_csv("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/FR_dic_search_aligned_ordered_5K.csv", sep=";", index=False)