import json
from collections import defaultdict
import pandas as pd
from collections import Counter
# Load the dataset

original_val='/Users/jianlu/Desktop/Testing_result/dataset/VQDv1/val.json'
def sampling(orginal,path=None): #replace the path with the path of the original dataset
    with open(original_val, 'r') as file:
        data = json.load(file)

    # Create a dictionary to hold the groups
    grouped_by_bounding_boxes = defaultdict(list)

    # Iterate through each entry in the dataset
    for entry in data:
        # Add entry to the corresponding group
        grouped_by_bounding_boxes[len(entry['gtbox'])].append(entry)

    # Separate entries with exactly zero and one bounding box
    single_box_entries = grouped_by_bounding_boxes.pop(1, [])

    # Convert the single box entries to a DataFrame for sampling
    df_single_box = pd.DataFrame(single_box_entries)

    # Proportional sampling function
    def proportional_sampling(df, sample_percentage, K):
        total_sample_size = int(sample_percentage * len(df))
        print(f"Total sample size: {total_sample_size}")
        group_sizes = df['question_type'].value_counts(normalize=True)
        
        def sample_group(group):
            required_samples = int(total_sample_size * group_sizes[group.name])
            if len(group) < K:  # case where the entire group is smaller than K, sample all
                return group.sample(len(group), replace=True)
            elif required_samples < K:  # case where the required samples(after multiply sample_percentage) are smaller than K, sample K
                return group.sample(K, replace=True)
            return group.sample(required_samples)  # all other cases, sample_percentage * group_size > K, sample required_samples

        sample = df.groupby('question_type').apply(sample_group).reset_index(drop=True)
        return sample

    # Set parameters for sampling
    sample_percentage = 0.1  # For example, 10%
    K = 200  # Minimum samples per group

    # Perform proportional sampling on entries with exactly one bounding box
    sampled_single_box_df = proportional_sampling(df_single_box, sample_percentage, K)

    # Convert the sampled DataFrame back to a list of dictionaries
    sampled_single_box_data = sampled_single_box_df.to_dict(orient='records')

    # Combine sampled single box entries with other groups
    combined_data = []
    for entries in grouped_by_bounding_boxes.values():
        combined_data.extend(entries)
    combined_data.extend(sampled_single_box_data)

    # Save the combined data to a new JSON file
    
    with open(path, 'w') as file:
        json.dump(combined_data, file, indent=4)



    question_types = [entry['question_type'] for entry in combined_data]

    # Count the occurrences of each question type
    question_type_distribution = Counter(question_types)

    # Print the distribution
    print("Distribution of question types:")
    for question_type, count in question_type_distribution.items():
        print(f"{question_type}: {count}")

    dicts = {}
    for a in combined_data:
        n = -1
        if a['gtbox'] == [[]]:
            n = 0
        else:
            n = len(a['gtbox'])
        if n not in dicts:
            dicts[n] = 0
        dicts[n] += 1
    
    for a in sorted(dicts):
        print(f"Number of bounding boxes: {a}, number of questions: {dicts[a]}")

    
import argparse
parser = argparse.ArgumentParser(description="Description of your program")

# Add the arguments
parser.add_argument('--target_path', type=str, help='answer_path')
args = parser.parse_args()
sampling(path=args.target_path)
