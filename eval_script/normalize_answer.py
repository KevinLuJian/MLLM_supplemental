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
    # path = Path(path)

    # # Get the directory and filename
    # dir_path = path.parent
    # filename = path.stem  # filename without extension
    # ext = path.suffix  # file extension

    # # Modify the filename
    # new_filename = filename + "_processed" + ext

    # # Combine the directory and new filename
    # new_path = dir_path / new_filename
    # with open(new_path, 'w') as f:
    #     for item in data:
    #         f.write(json.dumps(item) + '\n')

# import argparse
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', type=str, default='/')
#     args = parser.parse_args()

#     normalize(args.path)


