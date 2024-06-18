import pandas as pd
import json
import random
import os
# Load JSON data from a file
TDIUC_question = '/Users/jianlu/Desktop/Testing_result/dataset/TDIUC/OpenEnded_mscoco_val2014_questions.json'
TDIUC_annotation = '/Users/jianlu/Desktop/Testing_result/dataset/TDIUC/mscoco_val2014_annotations.json'
def merging(question, annotation):
    with open(question, 'r') as file:
        question = json.load(file)
    question = question['questions']
    with open(annotation, 'r') as file:
        annotation = json.load(file)
    annotation = annotation['annotations']
    '''
            {
        "index": 386870,
        "question_id": 10425545,
        "question": "The women is wear what color?",
        "image_id": 382374,
        "labeled_answer": "doesnotapply",
        "question_type": "absurd"
    },
    '''
    merge_data = []
    i = 0
    for q,a in zip(question, annotation):
        record = {
            "index": i,
            "question_id": q['question_id'],
            "question": q['question'],
            "image_id": q['image_id'],
            "labeled_answer": a['answers'][0]['answer'],
            "question_type": a['question_type']
        }
        i += 1
        merge_data.append(record)
    # with open('a.json', 'w') as file:
    #     json.dump(merge_data, file, indent=4)
    return merge_data



def sampling():
    data = merging(TDIUC_question, TDIUC_annotation)
    df = pd.DataFrame(data)

    sample_percentage = 0.05  # For example, 10%
    K = 200  # Minimum samples per group

    def proportional_sampling(df, sample_percentage, K):
        total_sample_size = int(sample_percentage * len(df))
        print(f"total_sample_size", total_sample_size)
        group_sizes = df['question_type'].value_counts(normalize=True)
        
        def sample_group(group):
            required_samples = int(total_sample_size * group_sizes[group.name])
            if len(df) * group_sizes[group.name]< K: # case where the entire group is smaller than K, sample all
                return group.sample(int(len(df) * group_sizes[group.name]))
            elif required_samples < K: # case where the required samples(after multiply sample_percentage) are smaller than K, sample K
                return group.sample(K)
        
            return group.sample(required_samples)# all other cases, sample_percentage * group_size > K, sample required_samples
        sample = df.groupby('question_type').apply(sample_group).reset_index(drop=True)
        return sample

    sample = proportional_sampling(df, sample_percentage, K)
    sample_dict = sample.to_dict(orient='records')
    # random.shuffle(sample_dict)
    return sample_dict




import argparse
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--target_path', type=str, help='answer_path')
args = parser.parse_args()
sample_dict = sampling()

with open(args.target_path, 'w') as file:
        json.dump(sample_dict, file, indent=4)
       

dicts = {}
with open(args.target_path, 'r') as file:
    datas = json.load(file)
    for a in datas:
        if a['question_type'] not in dicts:
            dicts[a['question_type']] = 0
        dicts[a['question_type']] += 1

for a in dicts:
    print(f"question type: {a}, total: {dicts[a]}")