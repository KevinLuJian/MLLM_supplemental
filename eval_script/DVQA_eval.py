import json
from num2words import num2words
from decimal import Decimal, InvalidOperation
import re
import numpy as np
'''
Format:       "question_id": 3113,
              "question": "What is the accuracy of the algorithm gloom in the dataset chaos?",
              "image_id": "bar_val_hard_00000001.png",
              "predicted_answer": "0.5",
              "labeled_answer": "5",
              "template_id": "data",
              "model_id": "llava-v1.5-7b"
'''
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

    data = open_jsonl_file(file_path)

   
    my_dict = {}


    overall_total = 0
    overall_correct = 0
    for a in data:

        overall_total += 1

        labeled_answer = a['labeled_answer']
        labeled_answer = labeled_answer.replace(' ', '').lower().replace('.', '')

        predicted_answer = a['predicted_answer']
        predicted_answer = predicted_answer.replace(' ', '').lower().replace('.', '')


        if predicted_answer.isdigit():
            try:
                predicted_answer = num2words(predicted_answer)
            except Exception as e: # handle the respond that is in superscript form, from GPT-4(Very Seldomly)
                predicted_answer = convert_superscript_to_number(predicted_answer)
                predicted_answer = num2words(predicted_answer)
        
        if labeled_answer.isdigit():
            labeled_answer = num2words(labeled_answer)

        
        #record the accuracy of each question type, template_id = {reasoning, structural, retrieval}
        if a['template_id'] not in my_dict:
            my_dict[a['template_id']] = (0,0)

        # retrieve the previous correct and total number of the question type
        correct, total = my_dict[a['template_id']]

        if predicted_answer == labeled_answer:
            my_dict[a['template_id']] = (1,1)
            correct += 1
            overall_correct += 1
        
        total += 1
        # update the correct and total number of the question type
        my_dict[a['template_id']] = (correct, total)
    

    macro_accuracy = 0
    for type in my_dict:
        correct, total = my_dict[type]
        print(f"question type: {type}: accuracy: {correct/total}, total {type} question = {total}, correct = {correct}")
        macro_accuracy += correct/total
    print(f"micro accuracy: {overall_correct/overall_total:.4f}")
    print(f"macro accuracy: {macro_accuracy/len(my_dict):.4f}")

import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add the arguments
    parser.add_argument('--path', type=str, help='The path to process')

    # Parse the arguments
    args = parser.parse_args()
    eval_model(args.path)