import json
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

def eval_model(path):
    data = []
    
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    i = 0
    overall_correct = 0
    my_dict = {}
    overall_total = 0
    
    for a in data:
        overall_total += 1
        label_answer = a['labeled_answer'].replace(' ', '').lower()
        predicted_answer = a['predicted_answer'].replace(' ','').lower()

        if predicted_answer.isdigit():
            predicted_answer = num2words(predicted_answer)
        if label_answer.isdigit():
            label_answer = num2words(label_answer)

        if a['question_type'] not in my_dict:
            my_dict[a['question_type']] = (0,0)

        correct, total = my_dict[a['question_type']]
        total += 1
        if predicted_answer == label_answer or are_synonyms(predicted_answer,label_answer):
            correct += 1
            overall_correct += 1

        my_dict[a['question_type']] = (correct,total)

    

    acc_sum = 0
    for q_type, (correct, total) in my_dict.items():
        accuracy = correct / total
        acc_sum += accuracy
        print(f"Question type: {q_type}: accuracy: {accuracy:.4f}, questions = {total}, correct = {correct}")

    
    arithmetic_mpt = acc_sum / 12
    print(f"Micro accuracy: {overall_correct/overall_total:.4f}, total question = {overall_total}, correct = {overall_correct}")
    print(f"Macro accuracy: {arithmetic_mpt:.4f}")
    # print(f"Arithmetic MPT: {arithmetic_mpt:.4f}")


import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your program")
    
    # Add the arguments
    parser.add_argument('--path', type=str, help='answer_path')
    # Parse the arguments
    args = parser.parse_args()
    eval_model(args.path)
