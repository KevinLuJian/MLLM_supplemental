import pandas as pd
import json
import random
import os
# Load JSON data from a file
DVQA_easy = '/Users/jianlu/Desktop/DVQA/val_easy_qa.json'
DVQA_hard = '/Users/jianlu/Desktop/DVQA/val_hard_qa.json'
def sampling(path):
    df = pd.read_json(path)

    sample_percentage = 0.025  # For example, 10%
    K = 200  # Minimum samples per group

    def proportional_sampling(df, sample_percentage, K):
        total_sample_size = int(sample_percentage * len(df))
        print(f"total_sample_size", total_sample_size)
        group_sizes = df['template_id'].value_counts(normalize=True)
        
        def sample_group(group):
            required_samples = int(total_sample_size * group_sizes[group.name])
            if len(df) * group_sizes[group.name]< K: # case where the entire group is smaller than K, sample all
                return group.sample(int(len(df) * group_sizes[group.name]))
            elif required_samples < K: # case where the required samples(after multiply sample_percentage) are smaller than K, sample K
                return group.sample(K)
        
            return group.sample(required_samples)# all other cases, sample_percentage * group_size > K, sample required_samples
        sample = df.groupby('template_id').apply(sample_group).reset_index(drop=True)
        return sample



    sample = proportional_sampling(df, sample_percentage, K)
    sample_dict = sample.to_dict(orient='records')
    random.shuffle(sample_dict)
    return sample_dict

    


import argparse
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--target_path', type=str, help='answer_path')
args = parser.parse_args()
sample_dict1 = sampling(path=DVQA_easy)
sample_dict2 = sampling(path=DVQA_hard)
sample_dict1.extend(sample_dict2)

with open(args.target_path, 'w') as file:
        json.dump(sample_dict1, file, indent=4)
       

dicts = {}
with open(args.target_path, 'r') as file:
    datas = json.load(file)
    for a in datas:
        if a['template_id'] not in dicts:
            dicts[a['template_id']] = 0
        dicts[a['template_id']] += 1

for a in dicts:
    print(f"question type: {a}, total: {dicts[a]}")