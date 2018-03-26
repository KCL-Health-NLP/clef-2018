import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle

###############################################
### read raw data from brutes csv
### (run on training, dev, test)
### - read Brutes CSV file
### - save distinct doc ids in external pickle
###############################################

# input csv file
lan = "FR"
folder = "T:/Natalia Viani/clef2018/training_CLEFeHealth2018_FR/FR/"
set_name = "training"
file_name = "CausesBrutes_FR_2006-2012"
doc=folder+"training/raw/corpus/"+file_name+".csv"

# read into a dataframe
df = pd.read_csv(doc,sep=';')

# fill empty csv values with "-1"
df = df.fillna(-1)

# list of columns to convert to int
scores=['DocID','YearCoded','LineID','IntType','IntValue']

# turn column types to numeric (so that they won't be hot-encoded automatically by get_dummies)
df[scores] = df[scores].apply(pd.to_numeric)

# extract unique document ids
DocID_unique = np.unique(df['DocID'])
print("number of documents:", len(DocID_unique)) 
print("number of lines:", len(df['LineID'])) 

# save document ids in an external file
pickle_out = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_"+set_name+"_docs.pickle","wb")
pickle.dump(DocID_unique, pickle_out)
pickle_out.close()

# save dataframe in csv file for manual check
df.to_csv('T:/Natalia Viani/clef2018/output_brutes.csv')