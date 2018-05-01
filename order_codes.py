# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:47:50 2018

@author: Natalia
"""
import pandas as pd
import pickle
import ast

#read input csv for dev file
folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
csv_file = folder+'training/aligned/corpus/string_matching_output/AlignedCauses_2006-2012full_string_matched.csv'
df_system = pd.read_csv(csv_file, sep=";") #sep can be , or ;
df_system = df_system.fillna("NULL")
#test case
#df_system = df_system.head(5000)

#read training aligned
folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
textf = folder+'training/aligned/corpus/AlignedCauses_2006-2012full.csv'
df_training = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 
df_training = df_training.fillna("NULL")

#compute code frequencies in training
df_training_icds = df_training.groupby("ICD10").size().reset_index(name="counts")

#order codes by frequency
df_training_icds = df_training_icds.sort_values(by=["counts"], ascending = False)
freq_list_sorted = df_training_icds["ICD10"].tolist()

#add new codes to the vector's tail (low frequencies)
for icd_codes in df_system["dictionary_lookup"]:
    icd_codes = icd_codes.replace("nan", "''") #to deal with nan cases, e.g. see line 4267: [nan, 'K746']
    icd_codes = ast.literal_eval(icd_codes)
    icd_codes = list(filter(None, icd_codes)) #to remove null codes
    for icd_code in icd_codes:
        if icd_code not in freq_list_sorted:
            freq_list_sorted.append(icd_code)

#create updated dataframe: there are probably better ways to do this   
df_output = pd.DataFrame(columns=['DocID', 'YearCoded', 'Gender', 'Age', 'LocationOfDeath', 'LineID', 'RawText', 'IntType', 'IntValue', 'CauseRank', 'StandardText', 'ICD10', 'dictionary_lookup', 'full_dictionary_lookup'])

index_output = 0
for index, row in df_system.iterrows():
    if ((index_output % 1000) == 0):
        print (index_output)
    icd_codes = row['dictionary_lookup']
    icd_codes = icd_codes.replace("nan", "''")
    icd_codes = ast.literal_eval(icd_codes)
    icd_codes = list(filter(None, icd_codes))
	#sort icd codes
    sorted_list = sorted(icd_codes, key=lambda x: freq_list_sorted.index(x))
    icd_texts = row['full_dictionary_lookup']
    df_output.loc[index_output] = [row['DocID'], row['YearCoded'], row['Gender'], row['Age'], row['LocationOfDeath'], row['LineID'], row['StandardText'], row['IntType'], row['IntValue'],row['CauseRank'], row['StandardText'], row['ICD10'], sorted_list,icd_texts]
    index_output=index_output+1
        
df_output.DocID = df_output.DocID.astype(int)
df_output.LineID = df_output.LineID.astype(int)
df_output.YearCoded = df_output.YearCoded.astype(int)

df_output.to_pickle("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/train_output_ordered.pickle")
df_output.to_csv("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/train_output_ordered.csv", sep=";", index=False)
