import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
import random

#############################
### split_data_calculees_FR
#############################

def create_dataset(available_ids, set_dim):
    set_indexes = random.sample(range(0,len(available_ids)-1), set_dim)
    set_indexes.sort()
    set_docs = available_ids[set_indexes]
    return set_docs, set_indexes

def compute_avg(set_name,df):
    codes_per_doc = df.groupby('DocID').ICD10.nunique()
    mean_num = np.mean(codes_per_doc)
    print(set_name," - avg number of codes per doc (including 'no code'):", mean_num)
    
def count_unique_codes(set_name, df):
    df_codes = df.loc[df['ICD10'] != '-1']
    codes_unique = np.unique(df_codes['ICD10'])
    print("unique codes in "+set_name+":", len(codes_unique))
    return codes_unique

def avg_unseen_codes(set_name, df, train_codes_unique, dev_codes_unique):
    unseen_codes = np.setdiff1d(dev_codes_unique, train_codes_unique)
    print("unseen codes in "+set_name+":", len(unseen_codes))    
    # create table with lines including unseen codes
    df_unseen_codes = df[df['ICD10'].isin(unseen_codes)]    
    # count avg number of unseen codes per doc
    unseen_codes_per_doc = df_unseen_codes.groupby('DocID').ICD10.nunique()
    mean_num_unseen = np.mean(unseen_codes_per_doc)
    print("avg number of unseen codes per doc:", mean_num_unseen)

# initialize random seed
random.seed(10)
    
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

# read doc ids from Brutes file
pickle_in = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_"+set_name+"_docs.pickle","rb")
all_docs = pickle.load(pickle_in)
pickle_in.close()

# take only docs available in Brutes file
df = df[df['DocID'].isin(all_docs)]
df['ICD10'] = df['ICD10'].astype('str')

df_docyear = df.groupby(['YearCoded','DocID']).size().reset_index(name="counts")

df_year = df_docyear.groupby('YearCoded').size().reset_index(name="counts")

# Year: 2009    Docs: 9299
count = df_year.loc[df_year['YearCoded'] == 2009]['counts'].iloc[0]
# take only docs in 2009
df_one_year = df[df['YearCoded']== 2009]
docs_one_year = np.unique(df_one_year['DocID'])

# dataset division
dev_num = int(count/2) #4649
test_num = int(count/2)+1 #4650

[test_docs, test_indexes] = create_dataset(docs_one_year, test_num)
dev_docs = np.delete(docs_one_year, test_indexes)
training_docs = np.unique(df[df['YearCoded']!= 2009]['DocID'])

# compute avg number of labels per doc
compute_avg("training", df[df['DocID'].isin(training_docs)])
compute_avg("dev", df[df['DocID'].isin(dev_docs)])
compute_avg("test", df[df['DocID'].isin(test_docs)])

# count unique labels
train_codes_unique=count_unique_codes("training", df[df['DocID'].isin(training_docs)])
dev_codes_unique=count_unique_codes("dev", df[df['DocID'].isin(dev_docs)])
test_codes_unique=count_unique_codes("test", df[df['DocID'].isin(test_docs)])

# count and average unseen labels
avg_unseen_codes("dev", df[df['DocID'].isin(dev_docs)], train_codes_unique, dev_codes_unique)
avg_unseen_codes("test", df[df['DocID'].isin(test_docs)], train_codes_unique, test_codes_unique)

training_year = []
for docid in training_docs:
    docyear = df[df['DocID']==docid]['YearCoded'].iloc[0]
    training_year.append(docyear)
    
# create final dataframe with required information
df_train_docs = pd.DataFrame(data={'DocID': training_docs, 'YearCoded':training_year, 'Set':'training'})
df_dev_docs = pd.DataFrame(data={'DocID': dev_docs, 'YearCoded':2009, 'Set':'dev'})
df_test_docs = pd.DataFrame(data={'DocID': test_docs, 'YearCoded':2009, 'Set':'test'})

df_final = pd.concat([df_train_docs, df_dev_docs, df_test_docs])
    
pickle_out = open("T:/Natalia Viani/clef2018/scripts/pickles/"+lan+"_internal_dataset_division.pickle","wb")
pickle.dump(df_final, pickle_out)
pickle_out.close()

# print to excel
writer = pd.ExcelWriter("T:/Natalia Viani/clef2018/"+lan+"_internal_dataset_division.xlsx")
df_final.to_excel(writer,'dataset_division')
writer.save()