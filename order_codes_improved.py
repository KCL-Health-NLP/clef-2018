# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:47:50 2018

@author: Natalia
"""
import pandas as pd
import pickle
import ast

folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
csv_file = folder+'training/aligned/corpus/string_matching_output/AlignedCauses_2006-2012full_string_matched.csv'
#csv_file = folder+'training/aligned/corpus/string_matching_output/AlignedCauses_2013fullcodes.csv'

#be careful with sep
df_system = pd.read_csv(csv_file, sep=";")
df_system = df_system.fillna("NULL")

#df_system = df_system.head(5000)
docIds = df_system["DocID"].tolist()

lan = "FR"
folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
## aligned
textf = folder+'training/aligned/corpus/AlignedCauses_2006-2012full.csv'
df_training = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 
df_training = df_training.fillna("NULL")

print(len(df_training))
df_training_icds = df_training.groupby("ICD10").size().reset_index(name="counts")
print(len(df_training_icds))

df_training_icds = df_training_icds.sort_values(by=["counts"], ascending = False)
freq_list_sorted = df_training_icds["ICD10"].tolist()

for icd_codes in df_system["dictionary_lookup"]:
    icd_codes = icd_codes.replace("nan", "''") #see line 4267: [nan, 'K746']
    icd_codes = ast.literal_eval(icd_codes)
    icd_codes = list(filter(None, icd_codes))
    for icd_code in icd_codes:
        if icd_code not in freq_list_sorted:
            freq_list_sorted.append(icd_code)

print(len(freq_list_sorted))

pickle_in = open("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/train_output_ordered_241K.pickle","rb")
df_first = pickle.load(pickle_in)
pickle_in.close()

last_row = df_first.tail(1)
#index: 180220
#remaining: 86587
df_system = df_system.tail(len(df_system)-len(df_first))


#df_output = pd.DataFrame(columns=['DocID', 'YearCoded', 'Gender', 'Age', 'LocationOfDeath', 'LineID', 'RawText', 'IntType', 'IntValue', 'CauseRank', 'StandardText', 'ICD10', 'dictionary_lookup', 'full_dictionary_lookup'])
df_output = df_first

dic = {}
index = 1
for code in freq_list_sorted:
    dic[code] = index
    index = index+1
    
#index_output = 0
index_output = len(df_output)
for index, row in df_system.iterrows():
    if ((index_output % 1000) == 0):
        print (index_output)
    icd_codes = row['dictionary_lookup']
    icd_codes = icd_codes.replace("nan", "''")
    icd_codes = ast.literal_eval(icd_codes)
    icd_codes = list(filter(None, icd_codes))
    #d2 = {position: code for code, position in dic.iteritems() if code in icd_codes}
    d2 = {}
    for cod in icd_codes:
        d2[dic[cod]]=cod
    d2_ordered = sorted(d2)
    sorted_list = []
    for position in d2_ordered:
        sorted_list.append(d2[position])
    #sorted_list = sorted(icd_codes, key=lambda x: freq_list_sorted.index(x))
    icd_texts = row['full_dictionary_lookup']
    df_output.loc[index_output] = [row['DocID'], row['YearCoded'], row['Gender'], row['Age'], row['LocationOfDeath'], row['LineID'], row['StandardText'], row['IntType'], row['IntValue'],row['CauseRank'], row['StandardText'], row['ICD10'], sorted_list,icd_texts]
    index_output=index_output+1
        
df_output.DocID = df_output.DocID.astype(int)
df_output.LineID = df_output.LineID.astype(int)
df_output.YearCoded = df_output.YearCoded.astype(int)

df_output.to_pickle("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/train_output_ordered.pickle")
df_output.to_csv("C:/Users/Natalia/Documents/KingsWork/CLEF-2018/train_output_ordered.csv", sep=";", index=False)
