'''
Builds new CSV files from the old SQL data, by linking patients
to their transcripts and diagnoses.

The output is a CSV file called 'combined_data.csv' that contains
all diagnoses that each patient has received.
'''

import pandas as pd
import numpy as np

from categorize_icd9 import *

def dense_dataframe(df, on):
    """ Combines entries of same value of a given column """
    lst = []
    for index, value in df.iteritems(): # loop through columns in the dataframe
        # combines rows of values of each column to a list, and store the lists into a list
        lst.append(df.groupby(on)[index].apply(list)) 
        
    processed_df = pd.concat(lst, axis=1) # convert the list of lists of values to a dataframe 
    processed_df[on] = processed_df[on].apply(lambda x:x[0]) # retrieve the same value of 'on' column from the list    
    return processed_df

def get_diagnoses(transcripts):
    """ Gets all of the diagnoses for a given patient
        Includes descriptions, ICD9 codes, acute?, start/end dates
    """
    diagnoses, descriptions_all, acute_all, icd9_all, start_all, stop_all = [], [], [], [], [], []

    # Iterates through each transcript to pull out diagnoses
    for i in transcripts:
        if i in conversion_dict1:
            diag = conversion_dict1[i]
            diagnoses.append(diag)

            descriptions_one, acute_one, icd9_one, start_one, stop_one = [], [], [], [], []
            for k in diag:
                descriptions_one.append(conversion_dict2['DiagnosisDescription'][k])
                acute_one.append(       conversion_dict2['Acute'][k])
                icd9_one.append(        conversion_dict2['ICD9Code'][k])
                start_one.append(       conversion_dict2['StartYear'][k])
                stop_one.append(        conversion_dict2['StopYear'][k])
            
            descriptions_all.append(descriptions_one)
            acute_all.append(acute_one)
            icd9_all.append(icd9_one)
            start_all.append(start_one)
            stop_all.append(stop_one)
            
    return diagnoses, descriptions_all, acute_all, icd9_all, start_all, stop_all

def combine_data(conversion, transcripts):
    """ Matches diagnosis data with each transcript """
    
    transcripts['CombinedData'] = transcripts.TranscriptGuid.apply(get_diagnoses)
    
    transcripts['DiagnosisGuid']        = [x[0] for x in transcripts['CombinedData']]
    transcripts['DiagnosisDescription'] = [x[1] for x in transcripts['CombinedData']]
    transcripts['Acute']                = [x[2] for x in transcripts['CombinedData']]
    transcripts['ICD9Code']             = [x[3] for x in transcripts['CombinedData']]
    transcripts['StartYear']            = [x[4] for x in transcripts['CombinedData']]
    transcripts['StopYear']             = [x[5] for x in transcripts['CombinedData']]
        
    return transcripts

def flatten_icd9(codes):
    """ Flattens nested arrays in the data frame """
    diags = []
    if len(codes) == 0:
        return
    
    for i in codes:
        for j in i:
            if len(j) != 0: diags.append(j)
                
    return diags



# Gathers all the relevant data
diagnosis_df = pd.read_csv('../../data/raw/trainingSet/training_SyncDiagnosis.csv')
transcript_df = pd.read_csv('../../data/raw/trainingSet/training_SyncTranscript.csv')
conversion_df = pd.read_csv('../../data/raw/trainingSet/training_SyncTranscriptDiagnosis.csv')

# Compresses the SQL data into dense data frames
processed_conversion_df = dense_dataframe(conversion_df, 'TranscriptGuid')
processed_transcripts_df = dense_dataframe(transcript_df, 'PatientGuid')

# Helper dictionaries
conversion_dict1 = processed_conversion_df.set_index('TranscriptGuid')['DiagnosisGuid'].to_dict()
conversion_dict2 = diagnosis_df.set_index('DiagnosisGuid').to_dict()

# Combines diagnoses with transcripts
combined_data = combine_data(processed_conversion_df, processed_transcripts_df)

# Gets all ICD9 codes for each patient
combined_data['ICD9CodeFlattened'] = combined_data.ICD9Code.apply(flatten_icd9)
combined_data['ICD9Binned'] = combined_data.ICD9CodeFlattened.apply(categorize_icd9)

# Assigns 0's or 1's to each patient if they have a diagnosis in their transcript (e.g. diabetes)
icd9_data = pd.get_dummies(combined_data.ICD9Binned.apply(pd.Series).stack()).sum(level=0).apply(np.sign)
icd9_data['PatientGuid'] = combined_data['PatientGuid']

icd9_data.to_csv('combined_data.csv')