import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle

########################################################################
### read_data_calculees_FR
### (run on training)
### - read Calculees CSV file and check if documents are the same listed in the Brutes CSV file (remove additional lines)
### - count lines with NO ICD code (optional)
### - find unique ICD codes and save them in external pickle
### - count distinct ICD codes (including the "no code" label) per document
### - compute average number of distinct ICD codes per document
#######################################################################
    
# input csv file
lan = "FR"
folder = "T:/Natalia Viani/clef2018/training_CLEFeHealth2018_FR/FR/"
set_name = "training"
file_name = "CausesCalculees_FR_2006-2012"
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
# 21 documents

# remove these additional documents
df = df[~df['DocID'].isin(additional_docs)]

# assign the str type to the ICD10 column
df['ICD10'] = df['ICD10'].astype('str')

# count rows with a non-empty ICD10 code
df_codes = df.loc[df['ICD10'] != '-1']
df_no_codes = df.loc[df['ICD10'] == '-1'] # 1091 rows with ICD10 = -1 (no code)

# count unique icd10 codes
icd10_codes_unique = np.unique(df_codes['ICD10']) # 3302 different ICD10 codes

# save training ICD10 codes
pickle_out = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_unique_codes.pickle","wb")
pickle.dump(icd10_codes_unique, pickle_out)
pickle_out.close()

## Turn ICD10 codes into numeric features with LabelEncoder
#le = preprocessing.LabelEncoder()
#le.fit(df['ICD10'])
#df['ICD10'] = le.transform(df['ICD10'])

#label_NO_code = le.transform(['-1'])[0] #label assigned to "no code" class

### av number of labels per doc
codes_per_doc = df.groupby('DocID').ICD10.nunique()
mean_num = np.mean(codes_per_doc)
print("avg number of codes per doc (including 'no code'):", mean_num)