import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
    
########################################################################
### count unseen_codes
### (run on dev, test)
### - read Calculees CSV file and check if documents are the same listed in the Brutes CSV file (remove additional lines)
### - count distinct ICD codes (including the "no code" label) per document
### - compute average number of distinct ICD codes per document
### - count unique unseen codes
### - compute average number of distinct unseen ICD codes per document
########################################################################

# input csv file
lan = "FR"
folder = "T:/Natalia Viani/clef2018/training_CLEFeHealth2018_FR/FR/"
set_name = "test"
file_name = "CausesCalculees_FR_2014"
doc=folder+"training/raw/corpus/"+file_name+".csv"

# read into a dataframe
df = pd.read_csv(doc,sep=';')

# fill empty csv values with "-1" (e.g. empty ICD code)
df = df.fillna(-1)

# list of columns to convert to int
scores=['DocID','YearCoded','LineID','Rank']

# turn column types to numeric (so that they won't be hot-encoded automatically by get_dummies)
df[scores] = df[scores].apply(pd.to_numeric)

# extract unique document ids
DocID_unique = np.unique(df['DocID'])

# read doc ids from Brutes file
pickle_in = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_"+set_name+"_docs.pickle","rb")
DocID_unique_brutes = pickle.load(pickle_in)
pickle_in.close()

# identify additional documents
additional_docs = list(set(DocID_unique) - set(DocID_unique_brutes))
# 3754 additional_docs for test, 0 additional_docs for dev

# remove these additional documents --- to check whether we need this
df = df[~df['DocID'].isin(additional_docs)]

# assign the str type to the ICD10 column
df['ICD10'] = df['ICD10'].astype('str') 

# count avg number of codes per doc
codes_per_doc = df.groupby('DocID').ICD10.nunique()
mean_num = np.mean(codes_per_doc)
print("avg number of codes per doc (including 'no code'):", mean_num)

# read icd10 codes from training set
pickle_in = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_unique_codes.pickle","rb")
train_codes_unique = pickle.load(pickle_in)
pickle_in.close()
print('unique codes in training:', len(train_codes_unique))

df_codes = df.loc[df['ICD10'] != '-1']
dev_codes_unique = np.unique(df_codes['ICD10'])
print("unique codes in "+set_name+":", len(dev_codes_unique))

unseen_codes = np.setdiff1d(dev_codes_unique,train_codes_unique)
print("unseen codes in "+set_name+":", len(unseen_codes))

# create table with lines including unseen codes
df_unseen_codes = df[df['ICD10'].isin(unseen_codes)]

# count avg number of unseen codes per doc
unseen_codes_per_doc = df_unseen_codes.groupby('DocID').ICD10.nunique()
mean_num_unseen = np.mean(unseen_codes_per_doc)
print("avg number of unseen codes per doc:", mean_num_unseen)