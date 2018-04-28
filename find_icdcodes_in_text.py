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



def get_string_ngrams(string):
    #log.info('creating ngrams from string')
    all_strings = []
    splitstring = nltk.word_tokenize(string.lower())
    #print(splitstring)
    ## find all ngrams 1-5 in length
    for i in range(1,5):
        tmp = [' '.join(n) for n in ngrams(splitstring,i)]
        if len(tmp)>0:
            all_strings.append(tmp)
    all_strings = list(itertools.chain.from_iterable(all_strings))
    all_strings = list(set(all_strings))     
    return all_strings

def get_code_string_dict(codedataframe):
    icdcodes = defaultdict()

    for x in range(len(codedataframe)):
        currentid = codedataframe.iloc[x,1]
        currentvalue = codedataframe.iloc[x,0]
        icdcodes.setdefault(currentid, [])
        ## ignore short terms and acronyms for now
        if len(currentvalue.lower().strip())>0:
            icdcodes[currentid].append(currentvalue.lower().strip())
    return icdcodes

def get_string_code_dict(codedataframe):
    icdcodes = defaultdict()

    ## create a dictionary with icdcodes and all associated strings ##
    for x in range(len(codedataframe)):
        currentid = codedataframe.iloc[x,1]
        currentvalue = codedataframe.iloc[x,0]
        icdcodes.setdefault(currentvalue.lower().strip(), [])
        icdcodes[currentvalue.lower().strip()].append(currentid)
    return icdcodes

def find_codes_in_text(rawstrings, icdcodes_strings):
    ## list to save all codes and put back in dataframe
    codes_to_add = []
    ## list also with strings associated with a code and found in a rawstring
    full_codes_to_add = []
    count = 0
    for rr in rawstrings:
        rr = rr.lower().strip()
        codes = []
        found_codes = []
        log.info(str(count)+' out of: '+str(len(rawstrings)))
        count+=1
        log.info("rawstring: "+rr)
        string_p = get_string_ngrams(rr)
        #print(string_p)
        ## loop over icdcodes
        for i in icdcodes_strings:
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
                        log.info('identical, saving: '+m+'('+i+')')
                        found_codes.append([m])
                        codes.append(i)
                        identical=True
                #print('intersect: '+rr+':'+str(list(intersect)))
                #print(i)
                if not identical:
                    found_codes.append(list(intersect))
                    codes.append(i)

        log.info('found strings in icdcodes: '+str(found_codes))
        log.info('found codes: '+str(codes))
        codes_to_add.append(codes)
        full_codes_to_add.append(found_codes)
    return codes_to_add, full_codes_to_add


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('code_string_matching')


    #### INPUT FILES #####
    #### These should of course not be hard coded... #####
    ## raw
    #df = pd.read_csv('/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/raw/corpus/CausesBrutes_FR_2006-2012.csv', sep=';')  
    ## aligned
    textf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/AlignedCauses_2006-2012full.csv'
    df = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True) 
    
    #### INPUT DICTIONARY FILES #####

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2006-2010.csv'
    df2 = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)  
    ### IMPORTANT NOTE: header missing in some dictionaries, have added manually 
    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2011.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2012.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2013.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2014.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)

    icdf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/Dictionnaire2015.csv'
    tmp = pd.read_csv(icdf, sep=';', error_bad_lines=False, warn_bad_lines=True)
    df2 = pd.concat([df2, tmp]).drop_duplicates().reset_index(drop=True)


    # test cases
    #df = df[df['RawText'].str.contains('thromboses veineuses')]
    #df2 = df2[df2['DiagnosisText'].str.contains('thrombose')] 
    #df = df.sample(200)

    ### Get the text from dataframe
    rawstrings = df['RawText'].tolist()  

    ### Generate a dictionary of icd codes and their associated strings from the dictionaries
    log.info('creating icd-code dictionary')
    icdcodes_strings = get_code_string_dict(df2)
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
    log.info('size of dataframe: '+str(len(df)))
    log.info('size of list of codes to add: '+str(len(to_add)))
    df['dictionary_lookup'] = to_add.values
    df['full_dictionary_lookup'] = more_to_add.values
    ## save file in same directory as input file, change extension from csv to pickle
    outf = textf.replace('csv', 'pickle')
    log.info('saving new dataframe in: '+outf)
    df.to_pickle(outf)
    
    t1 = time.time()
    log.info('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))

    ## Time info from old trie-approach
    ## time for 200 excluding short codestrings: 0:02:03.406197

