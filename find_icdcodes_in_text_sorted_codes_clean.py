# coding: utf-8
''' 
Script to search for icd codes (or any dictionary entries from file) in text - using ngrams and set intersection
Written by Sumithra Velupillai April 28th 2018
'''

import pandas as pd
import time
import datetime
import itertools
import sys
from collections import defaultdict
from nltk.util import ngrams
import nltk
import logging
import sys
import unicodedata

###create dictionary with frequencies###
def create_dic():
    textf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/AlignedCauses_2006-2012full.csv'
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

def get_string_ngrams(string, stemming=False):
    #log.info('creating ngrams from string')
    all_strings = []

    string = string.lower().strip()
    ## ignore diacritics
    string = ''.join((c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn'))
    splitstring = nltk.word_tokenize(string)


    if stemming:
        snowball_stemmer = nltk.stem.snowball.FrenchStemmer()
        splitstring = [snowball_stemmer.stem(w) for w in splitstring]
    #print(splitstring)
    ## find all ngrams 1-5 in length
    for i in range(1,5):
        tmp = [' '.join(n) for n in ngrams(splitstring,i)]
        if len(tmp)>0:
            all_strings.append(tmp)
    all_strings = list(itertools.chain.from_iterable(all_strings))
    all_strings = list(set(all_strings))     
    return all_strings

def get_code_string_dict(codedataframe, stemming=False):
    if stemming:
        snowball_stemmer = nltk.stem.snowball.FrenchStemmer()
    icdcodes = defaultdict()

    for x in range(len(codedataframe)):
        currentid = codedataframe.iloc[x,1]
        currentvalue = codedataframe.iloc[x,0]
        currentvalue = currentvalue.lower().strip()
        ## ignore diacritics
        currentvalue = ''.join((c for c in unicodedata.normalize('NFD', currentvalue) if unicodedata.category(c) != 'Mn'))
        splitstring = nltk.word_tokenize(currentvalue)
        if stemming:
            splitstring = [snowball_stemmer.stem(w) for w in splitstring]
        #print(' '.join(splitstring))
        icdcodes.setdefault(currentid, [])
        ## ignore short terms and acronyms for now
        #if len(currentvalue.lower().strip())>0:
        icdcodes[currentid].append(' '.join(splitstring))
    return icdcodes

#def get_string_code_dict(codedataframe):
#    icdcodes = defaultdict()
#
#    ## create a dictionary with icdcodes and all associated strings ##
#    for x in range(len(codedataframe)):
#        currentid = codedataframe.iloc[x,1]
#        currentvalue = codedataframe.iloc[x,0]
#        icdcodes.setdefault(currentvalue.lower().strip(), [])
#        icdcodes[currentvalue.lower().strip()].append(currentid)
#    return icdcodes

def find_codes_in_text(rawstrings, icdcodes_strings):
    log.info('searching for codes in text')
    ## list to save all codes and put back in dataframe
    codes_to_add = []
    ## list also with strings associated with a code and found in a rawstring
    full_codes_to_add = []
    count = 0
    t0 = time.time()
    for rr in rawstrings:
#        rr = rr.lower().strip()
#        rr = rr.upper().strip()
        codes = []
        found_codes = []
        if ((count % 1000) == 0):
            t1 = time.time()
            log.info('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))
            print (str(count)+' out of: '+str(len(rawstrings)))
            t0 = time.time()
        #log.info(str(count)+' out of: '+str(len(rawstrings)))
        count+=1
        #log.info("rawstring: "+rr)
        string_p = get_string_ngrams(rr, stemming=False)
        #print(string_p)
        ## loop over icdcodes
        for i in icdcodes_strings:
            identical = False
            #print(i)
            ## get all strings associated with this code
            codestrings = icdcodes_strings[i]
            ## check if any string is subset of rawstring by getting a set intersect
            intersect = (set(string_p) & set(codestrings))
            ## if the intersect is not an empty set, there was a match
            if len(intersect)>0:
                ## check if the two strings are identical, if so, no need to add a bunch of other matches
                identical = False
                for m in intersect:
                    if rr == m:
                        #log.info('identical, saving: '+m+'('+i+')')
                        found_codes.append([m])
                        codes.append(i)
                        identical=True
                #print('intersect: '+rr+':'+str(list(intersect)))
                #print(i)
                if not identical:
                    found_codes.append(list(intersect))
                    codes.append(i)

        #log.info('found strings in icdcodes: '+str(found_codes))
        #log.info('found codes: '+str(codes))
        codes_to_add.append(codes)
        full_codes_to_add.append(found_codes)
    return codes_to_add, full_codes_to_add


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

def print_with_list(df, training_code_freq_list, outfile):
    new_out = []
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)

        no = []
        no.append(v['DocID'])
        no.append(v['YearCoded'])
        no.append(v['Age'])
        no.append(v['Gender'])
        no.append(v['LocationOfDeath'])
        no.append(v['LineID'])
        no.append(v['RawText'])
        no.append(v['IntType'])
        no.append(v['IntValue'])
        no.append(v['CauseRank'])
        no.append(v['StandardText'])
        no.append(sorted_list)
        #print("original: "+str(clist))
        #print("sorted: "+str(sorted_list))
        
        new_out.append(no)
#    #print(new_out)
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','dictionary_lookup'])

    tmp2.to_csv(outfile,sep=';', index=False)

def print_line_by_line(df, training_code_freq_list, outfile):
    new_out = []
    print(len(df))
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)
        if len(sorted_list)>0:
            for c in sorted_list:
                no = []
                no.append(v['DocID'])
                no.append(v['YearCoded'])
                no.append(v['Age'])
                no.append(v['Gender'])
                no.append(v['LocationOfDeath'])
                no.append(v['LineID'])
                no.append(v['RawText'])
                no.append(v['IntType'])
                no.append(v['IntValue'])
                no.append(v['CauseRank'])
                no.append(v['StandardText'])
                no.append(c)
#            #print(str(c))
                new_out.append(no)
        else:
            no = []
            no.append(v['DocID'])
            no.append(v['YearCoded'])
            no.append(v['Age'])
            no.append(v['Gender'])
            no.append(v['LocationOfDeath'])
            no.append(v['LineID'])
            no.append(v['RawText'])
            no.append(v['IntType'])
            no.append(v['IntValue'])
            no.append(v['CauseRank'])
            no.append(v['StandardText'])
            no.append('NULL')
            new_out.append(no)
    print(len(new_out))
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])

    tmp2.to_csv(outfile,sep=';', index=False)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('code_string_matching')

    folder =  '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/'

    #### INPUT FILES #####
    #### These should of course not be hard coded... #####
    ## raw
    #df = pd.read_csv('/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/raw/corpus/CausesBrutes_FR_2006-2012.csv', sep=';')  
    ## aligned
    textf = folder+'AlignedCauses_2006-2012full.csv'
    ## aligned dev
    #textf = folder'AlignedCauses_2013full.csv'
    ## aligned test
    textf = folder+'AlignedCauses_2014_full.csv'
    df = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 
    
    #### INPUT DICTIONARY FILES #####

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2006-2010.csv'
    df2 = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)  
#    ### IMPORTANT NOTE: header missing in some dictionaries, have added manually 
    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2011.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)
#
#    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2012.csv'
#    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
#    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)
#
#    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2013.csv'
#    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
#    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)
#
    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2014.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
#    df2 = tmp
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)
#
#    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2015.csv'
#    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
#    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)


    # test cases
    #df = df[df['RawText'].str.contains('thromboses veineuses')]
    #df2 = df2[df2['DiagnosisText'].str.contains('thrombose')] 
    #df = df.sample(200)
    #df = df.head(5000)
    

    ### Get the text from dataframe
    rawstrings = df['RawText'].tolist()  

    ### Generate a dictionary of icd codes and their associated strings from the dictionaries
    log.info('creating icd-code dictionary')
    icdcodes_strings = get_code_string_dict(df2, stemming=False)
    #icdstrings_codes = get_string_code_dict(df2)

    ## simple test case ##
    #rawstrings = ['hypertension intracranienne']
    #rawstrings = ['cirrhose alccolique, arythmie cardiaque par fibrillation auriculaire, cardiopathie ischémique']

    
    t0 = time.time()
    ## find all codes and strings in text ##
    codes_to_add, full_codes_to_add = find_codes_in_text(rawstrings, icdcodes_strings)
    to_add = pd.Series(codes_to_add)
    #print(codes_to_add)

    ## add what was found to the original dataframe and save to a new pickle file ##
    more_to_add = pd.Series(full_codes_to_add)

    ## test eval ##
    tmp = df
    tmp['dictionary_lookup'] = to_add.values
    ## get frequency list from training data
    training_code_freq_list = create_dic()


    outf_s = folder+'string_matched/test_system_list.csv'
    outf_s2 = folder+'string_matched/test_system.csv'
    print_with_list(tmp, training_code_freq_list, outf_s)
    print_line_by_line(tmp, training_code_freq_list, outf_s2)
    # gold to new outfile
    outf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/test_gold.csv'
#    #outf = textf.replace('csv', 'pickle')
#    log.info('saving new dataframe in: '+outf)
    df[['DocID', 'YearCoded', 'Age', 'Gender', 'LocationOfDeath', 'LineID', 'RawText', 'IntType', 'IntValue', 'CauseRank', 'StandardText', 'ICD10']].to_csv(outf,sep=';', index=False)


#    log.info('size of dataframe: '+str(len(df)))
#    log.info('size of list of codes to add: '+str(len(to_add)))
#    df['dictionary_lookup'] = to_add.values
#    df['full_dictionary_lookup'] = more_to_add.values
#    ## save file in same directory as input file, change extension from csv to pickle
#    outf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/test_stemming.pickle'
#    #outf = textf.replace('csv', 'pickle')
#    log.info('saving new dataframe in: '+outf)
#    df.to_pickle(outf)
    
    t1 = time.time()
    log.info('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))

    ## Time info from old trie-approach
    ## time for 200 excluding short codestrings: 0:02:03.406197

