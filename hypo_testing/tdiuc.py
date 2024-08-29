import json
import numpy as np
import pandas as pd
from num2words import num2words
from nltk.corpus import wordnet as wn
import numpy as np

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if not synsets1 or not synsets2:
        return False
    for syn1 in synsets1:
        for syn2 in synsets2:
            if syn1 == syn2:
                return True
    return False

def open_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        # Parse the JSON file data into a dictionary
        for line in file:
            data.append(json.loads(line))
    return data

def eval_model(file_path):

    data = open_jsonl_file(file_path)
    correct_list = []
    q_type = []
    
    try:
        model_id = data[0]['model_id']
    except Exception as e:
        if 'gpt-4o' in file_path:
            model_id = 'GPT-4o'
        elif 'gpt-4-turbo' in file_path:
            model_id = 'GPT4-turbo'
        else:
            print(e)
            exit(1)
    x = 0
    for a in data:
        label_answer = a['labeled_answer'].replace(' ', '').lower()
        predicted_answer = a['predicted_answer'].replace(' ','').lower()

        if predicted_answer.isdigit():
            predicted_answer = num2words(predicted_answer)
        if label_answer.isdigit():
            label_answer = num2words(label_answer)
        if predicted_answer == label_answer or are_synonyms(predicted_answer,label_answer):
            correct_list.append(1)
        else:
            correct_list.append(0)
        q_type.append(a['question_type'])
    return correct_list, q_type,model_id
    


import os
import argparse



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Process multiple file paths.")

    # Add the --path argument that accepts multiple values
    parser.add_argument('--path', nargs='+', help='List of file paths to process')

    # Parse the arguments
    args = parser.parse_args()

    # Access the list of file paths
    file_paths = args.path
    
    correct_list = []
    q_list = None
    model_id_list = []
    for path in file_paths:
        print(f"Processing file: {path}")
        list, q_list, model_id = eval_model(path)
        correct_list.append(list)
        q_list = q_list
        model_id_list.append(model_id)
        
    
    
    data_type = ['Question_id'] + model_id_list + ['question_type']

    questions = []
    for i in range(len(correct_list[0])):
        questions.append(i + 1)

    data_dict = {
        'Question_id': questions,
    }
    for i in range(len(model_id_list)):
        data_dict[model_id_list[i]] = correct_list[i]
    data_dict['question_type'] = q_list
    df = pd.DataFrame(data_dict)
    
   
    df.to_csv('hypo_testing/tdiuc.csv', index=False)