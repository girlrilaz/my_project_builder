#!/usr/bin/env python
# coding: utf-8

# ###################################################################       
# #Script Name    :                                                                                              
# #Description    :                                                                                 
# #Args           :                                                                                           
# #Author         : Nikhil Rao in R, converted to Python by Nor Raymond                                              
# #Email          : nraymond@appen.com                                          
# ###################################################################

import os
import glob 
import pandas as pd
import numpy as np
import yaml
import warnings
from functools import reduce
warnings.filterwarnings("ignore")

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(config_path, config_name), 'r') as file:
        config = yaml.safe_load(file)

    return config

config_path = "conf/base"

try:
    
    # load yaml catalog configuration file
    config = load_config("catalog.yml")

    os.chdir(config["project_path"])
    root_path = os.getcwd()
    
except:
    
    os.chdir('..')
    # load yaml catalog configuration file
    config = load_config("catalog.yml")

    os.chdir(config["project_path"])
    root_path = os.getcwd()
    
# import data_processing module
import src.data.data_processing as data_processing

def language_selection(languages):

    while True:
        try:
            language_index = int(input("\nPlease select the number of the Language you are assessing: "))
            if language_index < min(languages.index) or language_index > max(languages.index):
                print(f"\nYou must enter numbers between {min(languages.index)} - {max(languages.index)}... Please try again")
                continue
            elif language_index == "":
                print("\nYou must enter any numbers")
                continue
            else:
                print(f"\nYou have selected {language_index} for {languages.iloc[language_index, 0]}")
                language_selected = languages.iloc[language_index, 0]
                break

        except ValueError:
            print(f"\nYou must enter numerical values only... Please try again")
            continue
        else:
            break
            
    return language_selected

def run_selection():

    run_value = str(input("\nPlease input the type of run eg. Deployment, Pilot 1, Pilot 2A ...etc.: "))
    print(f"\nRun type: {run_value}")
    
    return run_value


# #### Functions for Language Modification - getting the overall time taken


# function for Language Modification
def get_time_taken(df, language_selected):

    # Filter data based on selected language
    dfr = df[df['Language'] == language_selected]

    # Time Taken by Item
    dfr["Time_Taken_Seconds"] = (dfr['_created_at'] - dfr['_started_at']).dt.seconds

    # Time Taken Overall
    dfr_grouped = dfr.groupby('_worker_id').sum('Time_Taken_Seconds')
    dfr_grouped["Time_Taken_Minutes_Overall"] = dfr_grouped["Time_Taken_Seconds"] / 60
    dfr_grouped = dfr_grouped.reset_index()
    dfr = pd.merge(dfr, dfr_grouped[["Time_Taken_Minutes_Overall", "_worker_id"]], how = 'left', on = '_worker_id')

    return dfr

def get_time_taken_all(language_selected, rc, v1, v2):
    
    df_list = [rc, v1, v2]
    keys = ["rcR", "v1R", "v2R"]
    df_time = {}
    
    for df, key in zip(df_list, keys) :

        dfr = get_time_taken(df, language_selected)
        df_time[key] = dfr

    rcR, v1R, v2R = df_time["rcR"], df_time["v1R"], df_time["v2R"]    
    
    return rcR, v1R, v2R


# #### Functions for calculating Fail Rates

# #### REPORT 1 : "Near Exact Match" - v1_actual_correct_by_question

def v1_fail_rate(v1R):
    
    vR_temp = v1R[['Language', 'Market', '_worker_id', '_unit_id', 'question_', 'a_domain', 'a_register', 
                    'wordphrase_a', 'b_domain', 'b_register', 'wordphrase_b', 'difficulty', 'Answer', 'Score']]
    
    # first grouping
    vR_grouped = vR_temp.groupby(['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 
                                 'b_domain', 'b_register', 'wordphrase_b', 'difficulty', 'Answer', 'Score'])['_worker_id'].count().reset_index()
    vR_grouped = vR_grouped.rename(columns = {"_worker_id" : "Count_of_Test_Takers"})
    
    # second grouping
    vR_grouped['Total_Test_Takers'] = vR_grouped.groupby(['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 
                                    'b_domain', 'b_register', 'wordphrase_b', 'difficulty'])['Count_of_Test_Takers'].transform('sum')   
    vR_grouped['Fail_Rate'] = round((vR_grouped['Count_of_Test_Takers'] / vR_grouped['Total_Test_Takers']), 2)
    
    # filter Score 0 
    vR_grouped = vR_grouped[vR_grouped['Score'] == 0]
    
    # sort values by Market and Fail_rate descending 
    vR_grouped = vR_grouped.sort_values(['Market', 'Fail_Rate'], ascending=[True, False])
    
    vR_fail_rates = vR_grouped.reset_index(drop=True) #re-order df index
    
    return vR_fail_rates

def generate_report_1(v1R):
    
    v1_actual_correct_by_question = v1_fail_rate(v1R)
    
    return v1_actual_correct_by_question


# #### REPORT 2 : "Close Match" - v2_fail_rates

def v2_fail_rate(v2R):
    
    vR_temp = v2R[['Language', 'Market', '_worker_id', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                   'b_register', 'wordphrase_b', 'difficulty', 'Answers', 'Score']]
    
    # first grouping
    vR_grouped = vR_temp.groupby(['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                                  'b_register', 'wordphrase_b', 'difficulty', 'Answers', 'Score'])['_worker_id'].count().reset_index()
    vR_grouped = vR_grouped.rename(columns = {"_worker_id" : "Count_of_Test_Takers"})
    
    # second grouping
    vR_grouped['Total_Test_Takers'] = vR_grouped.groupby(['Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 
                                    'b_domain', 'b_register', 'wordphrase_b', 'difficulty'])['Count_of_Test_Takers'].transform('sum')   
    vR_grouped['Overall_Fail_Rate'] = round((vR_grouped['Count_of_Test_Takers'] / vR_grouped['Total_Test_Takers']), 2)
    
    # filter Score 0 
    vR_grouped = vR_grouped[vR_grouped['Score'] == 0]
    
    # sort values by Market and _unit_id 
    vR_grouped = vR_grouped.sort_values(['Market', '_unit_id'], ascending = [True, True])
    
    # drop Score column
    vR_grouped = vR_grouped.drop('Score', axis = 1)
    
    vR_fail_rates = vR_grouped.reset_index(drop=True) #re-order df index
    
    return vR_fail_rates

def v2_fail_rate_2(v2R):
    
    vR_temp = v2R[['Language', 'Market', '_worker_id', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                   'b_register', 'wordphrase_b', 'difficulty', 'rater_answer', 'Answers', 'Score']]
    
    # first grouping
    vR_grouped = vR_temp.groupby(['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                                  'b_register', 'wordphrase_b', 'difficulty', 'rater_answer', 'Answers', 'Score'])['_worker_id'].count().reset_index()
    vR_grouped = vR_grouped.rename(columns = {"_worker_id" : "Count_of_Test_Takers"})
    
    # second grouping
    vR_grouped['Total_Test_Takers'] = vR_grouped.groupby(['Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                                                          'b_register', 'wordphrase_b', 'difficulty'])['Count_of_Test_Takers'].transform('sum')   
    vR_grouped['Rate'] = round((vR_grouped['Count_of_Test_Takers'] / vR_grouped['Total_Test_Takers']), 2)
    
    # filter Score 0 
    vR_grouped = vR_grouped[vR_grouped['Score'] == 0]
    
    # sort values by Market and _unit_id 
    vR_grouped = vR_grouped.sort_values(['Market', '_unit_id', 'Score', 'Rate'], ascending = [True, True, True, False])
    
    # drop Score columns
    vR_grouped = vR_grouped.drop(['Score', 'Count_of_Test_Takers', 'Total_Test_Takers'], axis = 1)
    
    vR_fail_rates = vR_grouped.reset_index(drop=True) #re-order df index
    
    vR_fail_rates  = pd.pivot_table(vR_fail_rates, 
                           index=['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a', 'b_domain', 
                                  'b_register', 'wordphrase_b', 'difficulty', 'Answers'],
                           values='Rate', columns=['rater_answer']).reset_index()
    vR_fail_rates.columns.name = None # remove name for columns
    
    # remove duplicate rows in the dataframe
    vR_fail_rates = vR_fail_rates.drop_duplicates()
    
    return vR_fail_rates 

def merge_v2_fail_rates(v2_actual_correct_by_question, v2_actual_correct_by_question_with_answer):
    
    v2_fail_rates = pd.merge(v2_actual_correct_by_question_with_answer, v2_actual_correct_by_question, how = 'left', 
                            on = ["Language", "Market", "_unit_id", "question_", "a_domain", "a_register", "wordphrase_a", "b_domain",
                                  "b_register", "wordphrase_b", "difficulty", "Answers"])
    
    v2_fail_rates = v2_fail_rates[['Language', 'Market', '_unit_id', 'question_', 'a_domain', 'a_register', 'wordphrase_a',
                'b_domain', 'b_register', 'wordphrase_b', 'difficulty', 'Count_of_Test_Takers', 'Total_Test_Takers',
                'Overall_Fail_Rate', 'Answers', 'a_and_b_are_not_related', 'a_and_b_are_related', 'a_and_b_have_the_same_meaning',
                'a_is_more_specific_than_b', 'b_is_more_specific_than_a']]
    
    return v2_fail_rates

def generate_report_2(v2R):
    
    v2_actual_correct_by_question = v2_fail_rate(v2R)

    v2_actual_correct_by_question_with_answer = v2_fail_rate_2(v2R)

    v2_fail_rates = merge_v2_fail_rates(v2_actual_correct_by_question, v2_actual_correct_by_question_with_answer)
    
    return v2_fail_rates


# #### REPORT 3 : "Reading Comprehension" : rc_question_skill_pass_rate

def rc_fail_rate(rcR):

    vR_temp = rcR[['Language', '_worker_id', '_country', 'Market', 'Time_Taken_Seconds', '_unit_id', 'title', 'test_',
                'question_1_difficulty', 'question_1_google_translate_error', 'Question 1 Skill tested',
                'question_2_difficulty', 'question_2_google_translate_error', 'Question 2 Skill tested',
                'question_3_difficulty', 'question_3_google_translate_error', 'Question 3 Skill tested',
                'question_4_difficulty', 'question_4_google_translate_error', 'Question 4 Skill tested',
                'register', 'topic', 'text_type', 'complexity', 'familiarity', 
                'question_no_1', 'question_no_2', 'question_no_3', 'question_no_4',
                'Answer_no_1', 'Answer_no_2', 'Answer_no_3', 'Answer_no_4',
                'Score']]
    
    # evaluate if Answers are the same as the questions. If either Q or A are empty, return NaN
    if vR_temp['question_no_1'].isnull().all() == True or vR_temp['Answer_no_1'].isnull().all() == True:      
        vR_temp['a1'] = np.nan      
    else:   
        vR_temp['a1'] = np.where(vR_temp['question_no_1'] == vR_temp['Answer_no_1'], 1, 0).astype('str')
        
    if vR_temp['question_no_2'].isnull().all() == True or vR_temp['Answer_no_2'].isnull().all() == True:        
        vR_temp['a2'] = np.nan      
    else:       
        vR_temp['a2'] = np.where(vR_temp['question_no_2'] == vR_temp['Answer_no_2'], 1, 0).astype('str')      
        
    if vR_temp['question_no_3'].isnull().all() == True or vR_temp['Answer_no_3'].isnull().all() == True:  
        vR_temp['a3'] = np.nan 
    else:
        vR_temp['a3'] = np.where(vR_temp['question_no_3'] == vR_temp['Answer_no_3'], 1, 0).astype('str')
        
    if vR_temp['question_no_4'].isnull().all() == True or vR_temp['Answer_no_4'].isnull().all() == True:   
        vR_temp['a4'] = np.nan 
    else:
        vR_temp['a4'] = np.where(vR_temp['question_no_4'] == vR_temp['Answer_no_4'], 1, 0).astype('str')
    
    # Dropping columns
    vR_temp = vR_temp.drop(['question_no_1', 'question_no_2', 'question_no_3', 'question_no_4',
                            'Answer_no_1', 'Answer_no_2', 'Answer_no_3', 'Answer_no_4', 'Score'], axis =1)  
    
    # concatenate values from different columns with delimiter ;
    vR_temp['Score'] = vR_temp[['a1', 'a2', 'a3', 'a4']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Question'] = ';'.join(['Question 1', 'Question 2', 'Question 3', 'Question 4'])
    vR_temp['Difficulty'] = vR_temp[['question_1_difficulty', 'question_2_difficulty', 
                                     'question_3_difficulty', 'question_4_difficulty']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Google_Translate_Error'] = vR_temp[['question_1_google_translate_error', 
                                                 'question_2_google_translate_error', 
                                                 'question_3_google_translate_error', 
                                                 'question_4_google_translate_error']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Skill'] = vR_temp[['Question 1 Skill tested', 'Question 2 Skill tested', 
                                'Question 3 Skill tested', 'Question 4 Skill tested']].astype('str').agg(';'.join, axis=1) 
    
    # Dropping more columns
    vR_temp = vR_temp.drop(['question_1_difficulty', 'question_1_google_translate_error', 'Question 1 Skill tested', 
                            'question_2_difficulty', 'question_2_google_translate_error', 'Question 2 Skill tested',
                            'question_3_difficulty', 'question_3_google_translate_error', 'Question 3 Skill tested',
                            'question_4_difficulty', 'question_4_google_translate_error', 'Question 4 Skill tested',
                            'a1', 'a2', 'a3', 'a4'], axis =1)  
    
    # Python explode function to split delimited columns and expand to rows - row_separate in R
    vR_temp =  vR_temp.set_index(['Language', '_worker_id', '_country', 'Market', 'Time_Taken_Seconds',
       '_unit_id', 'title', 'test_', 'register', 'topic', 'text_type',
       'complexity', 'familiarity']).apply(lambda x: x.str.split(';').explode()).reset_index()
    
    vR_temp[['Score', 'Question', 'Difficulty', 'Google_Translate_Error', 'Skill']] = vR_temp[['Score', 'Question', 'Difficulty', 
                                                                                               'Google_Translate_Error', 'Skill']].replace('nan', np.nan)
    vR_temp = vR_temp.dropna(subset = ['Score'])  # remove rows with NaN values in Score 
    vR_temp['Score'] = vR_temp['Score'].astype('int') # set Score as integer
    
    rc_answer = vR_temp
    
    return vR_temp

## Melt RC and categorize question choice with letter and question number
def melt_rc_assign(rc_choices, q_list, choice_list):
    
    df=[]
    for ql in q_list:
        for cl in choice_list:
            df_temp_1 = rc_choices[rc_choices['variable'].str.contains('question_' + str(ql))]
            df_temp_2 = df_temp_1[df_temp_1['variable'].str.contains('choice_' + str(cl))]
            df_temp_2['Question'] = 'Question ' + str(ql)
            if cl == 1 :
                df_temp_2['Answer'] = 'a'
            elif cl == 2 :
                df_temp_2['Answer'] = 'b'
            elif cl == 3 :
                df_temp_2['Answer'] = 'c'
            df.append(df_temp_2)
            
    rc_choices = pd.concat(df)
    return rc_choices

## Melt RC and categorize question choice with letter and question number
def melt_rc(rcR):

    vR_temp = rcR[['Language', '_unit_id', 'title', 'test_',
                'question_1_choice_1', 'question_1_choice_2', 'question_1_choice_3',
                'question_2_choice_1', 'question_2_choice_2', 'question_2_choice_3',
                'question_3_choice_1', 'question_3_choice_2', 'question_3_choice_3',
                'question_4_choice_1', 'question_4_choice_2', 'question_4_choice_3']]
    
    # remove duplicate rows in the dataframe
    vR_temp = vR_temp.drop_duplicates().reset_index(drop=True)
    
    vR_temp = pd.melt(vR_temp, id_vars=['Language', '_unit_id', 'title', 'test_'])
    
    rc_choices = vR_temp
    
    q_list, choice_list = [1,2,3,4], [1,2,3]
    rc_choices = melt_rc_assign(rc_choices, q_list, choice_list)
    rc_choices = rc_choices[['Language', '_unit_id', 'title', 'test_', 'Question', 'Answer', 'variable', 'value']]
    rc_choices = rc_choices.sort_values(['Language', 'title', 'test_', 'Question', 'Answer'])
    
    actual_answer = rc_choices
    rater_answer = rc_choices
    
    return rc_choices, actual_answer, rater_answer

# ## Melt RC into long format with actual answers
def melt_rc_answer_actual(rcR):
    
    vR_temp = rcR[['Language', '_worker_id', '_country', 'Market', 'Time_Taken_Seconds', '_unit_id', 'title', 'test_',
                'question_1_difficulty', 'question_1_google_translate_error', 'Question 1 Skill tested',
                'question_2_difficulty', 'question_2_google_translate_error', 'Question 2 Skill tested',
                'question_3_difficulty', 'question_3_google_translate_error', 'Question 3 Skill tested',
                'question_4_difficulty', 'question_4_google_translate_error', 'Question 4 Skill tested',
                'register', 'topic', 'text_type', 'complexity', 'familiarity',
                'question_no_1', 'question_no_2', 'question_no_3', 'question_no_4',
                'Answer_no_1', 'Answer_no_2', 'Answer_no_3', 'Answer_no_4',
                'Score']]
    
    # evaluate if Answers are the same as the questions. If either Q or A are empty, return NaN
    if vR_temp['question_no_1'].isnull().all() == True or vR_temp['Answer_no_1'].isnull().all() == True:      
        vR_temp['a1'] = np.nan      
    else:   
        vR_temp['a1'] = np.where(vR_temp['question_no_1'] == vR_temp['Answer_no_1'], 1, 0).astype('str')
        
    if vR_temp['question_no_2'].isnull().all() == True or vR_temp['Answer_no_2'].isnull().all() == True:        
        vR_temp['a2'] = np.nan      
    else:       
        vR_temp['a2'] = np.where(vR_temp['question_no_2'] == vR_temp['Answer_no_2'], 1, 0).astype('str')      
        
    if vR_temp['question_no_3'].isnull().all() == True or vR_temp['Answer_no_3'].isnull().all() == True:  
        vR_temp['a3'] = np.nan 
    else:
        vR_temp['a3'] = np.where(vR_temp['question_no_3'] == vR_temp['Answer_no_3'], 1, 0).astype('str')
        
    if vR_temp['question_no_4'].isnull().all() == True or vR_temp['Answer_no_4'].isnull().all() == True:   
        vR_temp['a4'] = np.nan 
    else:
        vR_temp['a4'] = np.where(vR_temp['question_no_4'] == vR_temp['Answer_no_4'], 1, 0).astype('str')
    
    vR_temp = vR_temp.drop('Score', axis = 1)
    
    # concatenate values from different columns with delimiter ;
    vR_temp['Score'] = vR_temp[['a1', 'a2', 'a3', 'a4']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Rater_Answer'] = vR_temp[['question_no_1', 'question_no_2', 'question_no_3', 'question_no_4']].astype('str').agg(';'.join, axis=1)
    vR_temp['Actual_Answer'] = vR_temp[['Answer_no_1', 'Answer_no_2', 'Answer_no_3', 'Answer_no_4']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Question'] = ';'.join(['Question 1', 'Question 2', 'Question 3', 'Question 4'])
    vR_temp['Difficulty'] = vR_temp[['question_1_difficulty', 'question_2_difficulty', 
                                     'question_3_difficulty', 'question_4_difficulty']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Google_Translate_Error'] = vR_temp[['question_1_google_translate_error', 
                                                 'question_2_google_translate_error', 
                                                 'question_3_google_translate_error', 
                                                 'question_4_google_translate_error']].astype('str').agg(';'.join, axis=1) 
    vR_temp['Skill'] = vR_temp[['Question 1 Skill tested', 'Question 2 Skill tested', 
                                'Question 3 Skill tested', 'Question 4 Skill tested']].astype('str').agg(';'.join, axis=1) 
    
    vR_temp = vR_temp.drop(['question_1_difficulty', 'question_1_google_translate_error', 'Question 1 Skill tested', 
                            'question_2_difficulty', 'question_2_google_translate_error', 'Question 2 Skill tested',
                            'question_3_difficulty', 'question_3_google_translate_error', 'Question 3 Skill tested',
                            'question_4_difficulty', 'question_4_google_translate_error', 'Question 4 Skill tested',
                            'question_no_1', 'question_no_2', 'question_no_3', 'question_no_4',
                            'Answer_no_1', 'Answer_no_2', 'Answer_no_3', 'Answer_no_4',
                            'a1', 'a2', 'a3', 'a4'], axis = 1)
    
     # Python explode function to split delimited columns and expand to rows - row_separate in R
    vR_temp =  vR_temp.set_index(['Language', '_worker_id', '_country', 'Market', 'Time_Taken_Seconds',
       '_unit_id', 'title', 'test_', 'register', 'topic', 'text_type',
       'complexity', 'familiarity']).apply(lambda x: x.str.split(';').explode()).reset_index()
    
    vR_temp[['Score', 'Rater_Answer', 'Actual_Answer', 'Question', 'Difficulty', 'Google_Translate_Error', 'Skill']] = vR_temp[['Score', 'Rater_Answer', 
                                                                                                                                'Actual_Answer','Question', 
                                                                                                                                'Difficulty', 
                                                                                                                                'Google_Translate_Error', 
                                                                                                                                'Skill']].replace('nan', np.nan)
    vR_temp = vR_temp.dropna(subset = ['Score'])  # remove rows with NaN values in Score 
    vR_temp['Score'] = vR_temp['Score'].astype('int') # set Score as integer
    
    rc_answer_actual = vR_temp
    
    return rc_answer_actual

def rc_q_s_pass_rate(rc_answer):
    
    # first grouping
    vR_grouped = rc_answer.groupby(['Language', 'Market', '_unit_id', 'title', 'test_', 'Score', 'Question', 'Difficulty', 'register', 'Skill'])['_worker_id'].count().reset_index()
    vR_grouped = vR_grouped.rename(columns = {"_worker_id" : "Count"})
    
    # second grouping
    vR_grouped['Total'] = vR_grouped.groupby(['Language', 'Market', '_unit_id', 'title', 'test_', 'Question', 'Difficulty', 'register', 'Skill'])['Count'].transform('sum')   
    vR_grouped['Fail_Rate'] = round((vR_grouped['Count'] / vR_grouped['Total']), 2)
    
    # filter Score 0 
    vR_grouped = vR_grouped[vR_grouped['Score'] == 0]
    
    # sort values by Market and _unit_id 
    vR_grouped = vR_grouped.sort_values(['Market', 'Fail_Rate'], ascending = [True, False])
    vR_grouped = vR_grouped.reset_index(drop=True) #re-order df index
    
    rc_question_skill_pass_rate = vR_grouped
    
    return rc_question_skill_pass_rate

def generate_report_3(rcR):
    
    rc_answer = rc_fail_rate(rcR)
    
    rc_choices, actual_answer, rater_answer = melt_rc(rcR)
    
    rc_answer_actual = melt_rc_answer_actual(rcR)
    
    rc_question_skill_pass_rate = rc_q_s_pass_rate(rc_answer)
    
    return rc_question_skill_pass_rate


# #### REPORT 4 : "RC with Answers" : rc_question_skill_pass_rate_answer_final

def rc_q_s_pass_rate_answer(rc_answer_actual):
    
    # first grouping
    vR_grouped = rc_answer_actual.groupby(['Language', 'Market', '_unit_id', 'title', 'test_', 'Actual_Answer', 'Rater_Answer', 
                                    'Score', 'Question', 'Difficulty', 'register', 'Skill'])['_worker_id'].count().reset_index()
    vR_grouped = vR_grouped.rename(columns = {"_worker_id" : "Count"})
    
    # second grouping
    vR_grouped['Total'] = vR_grouped.groupby(['Language', 'Market', '_unit_id', 'title', 'test_', 'Question', 'Difficulty', 'register', 'Skill'])['Count'].transform('sum')   
    vR_grouped['Fail_Rate'] = round((vR_grouped['Count'] / vR_grouped['Total']), 2)
    
    # filter Score 0 
    vR_grouped = vR_grouped[vR_grouped['Score'] == 0]
    
    # sort values by Market and _unit_id 
    vR_grouped = vR_grouped.sort_values(['Market', '_unit_id', 'Question', 'Fail_Rate'], ascending = [True, True, True, False])
    vR_grouped = vR_grouped.reset_index(drop=True) #re-order df index
    
    rc_question_skill_pass_rate_answer = vR_grouped
    
    return rc_question_skill_pass_rate_answer

def join_rc_q_s_pass_rate_answer(rc_question_skill_pass_rate_answer, actual_answer, rater_answer):
    
    first_join = rc_question_skill_pass_rate_answer
    first_join = pd.merge(first_join, actual_answer, how = 'left', 
                            left_on = ["Language", "_unit_id", "title" , "test_", "Question", "Actual_Answer"],
                            right_on = ["Language", "_unit_id", "title" , "test_", "Question", "Answer"])
    first_join = first_join.drop('Answer', axis=1)
    
    second_join = pd.merge(first_join, rater_answer, how = 'left', 
                            left_on = ["Language", "_unit_id", "title" , "test_", "Question", "Rater_Answer"],
                            right_on = ["Language", "_unit_id", "title" , "test_", "Question", "Answer"])
    second_join = second_join.drop('Answer', axis=1)
    
    second_join = second_join[['Language', 'Market', '_unit_id', 'title', 'test_', 'Difficulty', 'register', 'Skill', 'Question',
                               'Actual_Answer', 'value_x', 'Rater_Answer', 'value_y', 'Count', 'Total', 'Fail_Rate']]
  
    second_join = second_join.rename(columns = { "Actual_Answer" : "Actual_Answer_Letter", 
                                       "value_x" : "Actual_Answer_Text",
                                       "Rater_Answer" : "Rater_Answer_Letter",
                                       "value_y" : "Rater_Answer_Text"})

    rc_question_skill_pass_rate_answer_final = second_join
    
    return rc_question_skill_pass_rate_answer_final


def generate_report_4(rcR):
    
    rc_choices, actual_answer, rater_answer = melt_rc(rcR)
    
    rc_answer_actual = melt_rc_answer_actual(rcR)
    
    rc_question_skill_pass_rate_answer = rc_q_s_pass_rate_answer(rc_answer_actual)

    rc_question_skill_pass_rate_answer_final = join_rc_q_s_pass_rate_answer(rc_question_skill_pass_rate_answer, actual_answer, rater_answer)
    
    return rc_question_skill_pass_rate_answer_final

# Functions to generate all reports

def generate_all_fail_rate_reports(rcR, v1R, v2R):
    
    # Report 1 - Near Exact Match - v1_actual_correct_by_question
    v1_actual_correct_by_question =  generate_report_1(v1R)

    # Report 2 - Close Match - v2_fail_rates
    v2_fail_rates = generate_report_2(v2R)
    
    # Report 3 - Reading Comprehension - rc_question_skill_pass_rate
    rc_question_skill_pass_rate = generate_report_3(rcR)
    
    # Report 4 - RC with Answers - rc_question_skill_pass_rate_answer_final
    rc_question_skill_pass_rate_answer_final = generate_report_4(rcR)
    
    # store all 4 reports into a dictionary set
    list_of_datasets = {"Near Exact Match" : v1_actual_correct_by_question,
                        "Close Match" : v2_fail_rates,
                        "Reading Comprehension" : rc_question_skill_pass_rate,
                        "RC with Answers" : rc_question_skill_pass_rate_answer_final}
    
    return list_of_datasets

def file_check_create(root_path, config, language_selected, run_value):
    
    run_folder = os.path.join(root_path, config['report']['deliverable'], run_value, language_selected)

    if not os.path.exists(run_folder):
        os.makedirs(run_folder, exist_ok=True)
        
    return run_folder

def write_fail_report_to_excel(run_folder, list_of_datasets, encode=None):
    
    with pd.ExcelWriter(os.path.join(run_folder, 'language_fail_rates.xlsx')) as writer:  
        for key, value in list_of_datasets.items():
            value.to_excel(writer, sheet_name=key, index=False, encoding=encode)


# #### Run all 

def main():
    
    print('\nData processing in progress...')
    # import data from data_processing module
    raters, r1, r2, r3, languages, rc, v1, v2 = data_processing.main()
    print('Data processing completed.')
    print("\n")
    print(languages)
    
    # Get input language selection
    language_selected = language_selection(languages)
    
    # Get input of run type
    run_value = run_selection()
    
    # Get data from language modification processes
    rcR, v1R, v2R = get_time_taken_all(language_selected, rc, v1, v2)
    
    print('\nGenerating language fail rates report...')
    
    # Start generating fail rate reports
    list_of_datasets = generate_all_fail_rate_reports(rcR, v1R, v2R)
    
    # Check the run type and language and create folders in reports > deliverables
    run_folder = file_check_create(root_path, config, language_selected, run_value)
    
    # Write reports to excel file in run_folder path
    write_fail_report_to_excel(run_folder, list_of_datasets, encode=None)
    
    print(f"\nLanguage fail rates report completed and stored in reports > deliverables > {run_value} > {language_selected}")

if __name__ == "__main__":
     
    main()