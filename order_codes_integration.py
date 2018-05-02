import pandas as pd
import ast

###create dictionary with frequencies###
def create_dic():
    folder = "C:/Users/Natalia/Documents/KingsWork/CLEF-2018/training_CLEFeHealth2018_FR/FR/"
    textf = folder+'training/aligned/corpus/AlignedCauses_2006-2012full.csv'
    df_training = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 
    df_training = df_training.fillna("NULL")
    df_training_icds = df_training.groupby("ICD10").size().reset_index(name="counts")
    df_training_icds = df_training_icds.sort_values(by=["counts"], ascending = False)    
    #dictionary with codes from training (ordered by counts)
    dic={}
    pos=1
    for index, row in df_training_icds.iterrows():
        code = row["ICD10"]
        dic[code] = pos
        pos = pos+1
    return dic


#### for each line in output ####
def order_line(dic, icd_codes):
    #create small dic
    d2 = {}
    unseen = []
    for cod in icd_codes:
    		if cod in dic:
    			d2[dic[cod]]=cod
    		else:
    			unseen.append(cod)
    #create ordered list
    d2_ordered = sorted(d2)
    sorted_list = []
    for position in d2_ordered:
        sorted_list.append(d2[position])     
    for cod in unseen:   
        sorted_list.append(cod) 
    return sorted_list
    
dic = create_dic()
icd_codes = "['ABC','I10', 'R092', 'X09', 'CDE']"
icd_codes = icd_codes.replace("nan", "''")
icd_codes = ast.literal_eval(icd_codes)
icd_codes = list(filter(None, icd_codes))
sorted_list = order_line(dic, icd_codes)
