import json
from num2words import num2words
from decimal import Decimal, InvalidOperation
import re
import numpy as np
import pandas as pd
'''
Format:       "question_id": 3113,
              "question": "What is the accuracy of the algorithm gloom in the dataset chaos?",
              "image_id": "bar_val_hard_00000001.png",
              "predicted_answer": "0.5",
              "labeled_answer": "5",
              "template_id": "data",
              "model_id": "llava-v1.5-7b"
'''
import json
from pathlib import Path
def word_to_digit(word):
    number_words = {
        'zero': 0,
        'none': 0,
        'no': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
    }
    
    return number_words.get(word, None)

def normalize(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    i = 0
    for a in data:
        answer = a['predicted_answer']
        answer = answer.replace('.','').replace(' ','').lower()
        if answer.isnumeric():
            a['predicted_answer'] = int(answer)
            # print(f"{i} :{answer}")
        else:
            if word_to_digit(answer) is not None:
                a['predicted_answer'] = word_to_digit(answer)
        i += 1
    return data

# very seldomly, but sometimes the predicted answer is in superscript form.(Only a few in GPT-4 responds)
def convert_superscript_to_number(exp_str):
    # Define a mapping of superscript characters to their normal digit counterparts
    superscript_map = {
        '⁰': 0, '¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9
    }
    
    # Regular expression to match the pattern "10" followed by superscript digits
    pattern = re.compile(r'^10[⁰¹²³⁴⁵⁶⁷⁸⁹]+$')
    
    if pattern.match(exp_str):
        # Separate the base ("10") from the superscript part
        base = 10
        exponent_str = exp_str[2:]

        # Convert the superscript part to a normal integer
        exponent = 0
        for char in exponent_str:
            exponent = exponent * 10 + superscript_map[char]

        # Calculate the result
        result = base ** exponent
        return result
    else:
        print(f"The string '{exp_str}' is not in the form '10' followed by superscript digits.")
        return None
def open_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        # Parse the JSON file data into a dictionary
        for line in file:
            data.append(json.loads(line))
    return data

def eval_model(file_path):

    data = normalize(file_path)
    correct_list = []
    q_type = []
    

    try:
        model_id = data[0]['model_id']
    except Exception as e:
        if 'gpt-4o' in file_path:
            model_id = 'GPT-4o'
        elif 'GPT4-turbo' in file_path:
            model_id = 'GPT4-turbo'
        elif 'LLaVA_1.5_7b' in file_path:
            model_id = 'llava-v1.5-7b'
        else:
            print(e)
            exit(1)
    x = 0
    try:
        for a in data:
            
            x += 1
            labeled_answer = a['labeled_answer']

            predicted_answer = a['predicted_answer']

        
            if predicted_answer == labeled_answer:
                correct_list.append(1)
            else:
                correct_list.append(0)

            q_type.append(a['issimple'])
    except Exception as e:
        print(f"Error in {file_path}")
        print(f"{x}: {a.keys()}")
        exit(1)
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
        
    
    
    data_type = ['Question_id'] + model_id_list + ['issimple']

    questions = []
    for i in range(len(correct_list[0])):
        questions.append(i + 1)

    data_dict = {
        'Question_id': questions,
    }
    for i in range(len(model_id_list)):
        data_dict[model_id_list[i]] = correct_list[i]
    data_dict['issimple'] = q_list
    df = pd.DataFrame(data_dict)
    
   
    df.to_csv('hypo_testing/tallyqa.csv', index=False)