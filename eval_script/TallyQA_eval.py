import json
import argparse
import numpy as np
from normalize_answer import normalize
def micro_analyse(answer_path):
    data = []
    data = normalize(answer_path)

    totaln = 0
    totalc = 0
    my_dict = {}
    RMSE = {}
    test = []
    for a in data:
        totaln += 1
        labeled_answer = a['labeled_answer']
        predicted_answer = a['predicted_answer']
        
        if a['issimple'] not in my_dict:
            if predicted_answer == labeled_answer:
                test.append(1)
                my_dict[a['issimple']] = (1, 1)
                totalc += 1
            else:
                my_dict[a['issimple']] = (0, 1)
                test.append(0)
        else:
            correct, total = my_dict[a['issimple']]
            if predicted_answer == labeled_answer:
                totalc += 1
                correct += 1
                test.append(1)
            else:
                test.append(0)
                pass
            total += 1
            my_dict[a['issimple']] = (correct, total)
        
        #calculate RMSE:
        if a['issimple'] not in RMSE:
            RMSE[a['issimple']] = []
        if not isinstance(predicted_answer, (int, float)):
            continue

        RMSE[a['issimple']].append((float(min(predicted_answer,15)) - float(labeled_answer))**2)
    for type, (correct, total) in my_dict.items():
        if type == True:
            print(f"Question Type: Simple Counting, Micro accuracy: {correct / total}, total {type} questions = {total}, correct = {correct}")
        else:
            print(f"Question Type: Complex Counting, Micro accuracy: {correct / total}, total {type} questions = {total}, correct = {correct}")
    print(f"")
    for type, errors in RMSE.items():
        if type == True:
            print(f"question type: Simple Counting: RMSE: {sum(errors) / len(errors)}")
        else:
            print(f"question type: Complex Counting: RMSE: {sum(errors) / len(errors)}")
            
   
def macro_accuracy(answer_path):
    data = normalize(answer_path)

    results = {}

    for entry in data:
        label_answer = entry['labeled_answer']
        predicted_answer = entry['predicted_answer']

        question_type = entry['issimple']

        if question_type not in results:
            results[question_type] = {}

        if label_answer not in results[question_type]:
            results[question_type][label_answer] = {'correct': 0, 'total': 0}

        if predicted_answer == label_answer:
            results[question_type][label_answer]['correct'] += 1

        results[question_type][label_answer]['total'] += 1
        
    normalized_results = {}
    for question_type, answers in results.items():
        normalized_results[question_type] = {}
        for answer, stats in answers.items():
            if answer <= 3:
                normalized_results[question_type][str(answer)] = {'correct': stats['correct'], 'total': stats['total']}
            else:
                if 'above' not in normalized_results[question_type]:
                    normalized_results[question_type]['above'] = {'correct': stats['correct'], 'total': stats['total']}
                else:
                    normalized_results[question_type]['above']['correct'] += stats['correct']
                    normalized_results[question_type]['above']['total'] += stats['total']


    for question_type, answers in normalized_results.items():
        if question_type == True:
            print(f"Question Type: Simple Counting")
        else:
            print(f"Question Type: Complex Counting")

        average_accuracy = 0
        for answer, stats in answers.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] != 0 else 0
            average_accuracy += accuracy
            print(f"Answer: {answer}, Accuracy: {accuracy:.4f}, Total: {stats['total']}, Correct: {stats['correct']}")
        print(f"Macro Accuracy: {average_accuracy / len(answers):.4f}")
              
# Example usage:
import os
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Description of your program")

    # Add the arguments
    parser.add_argument('--path', type=str, help='answer_path')
    # Parse the arguments
    args = parser.parse_args()
    print(f"====================TallyQA Evaluations {args.path}====================\n")
    print(f"MICRO")
    micro_analyse(args.path)
    print(f"MACRO")
    macro_accuracy(args.path)
    
