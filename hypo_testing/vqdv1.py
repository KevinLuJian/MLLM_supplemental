import re
import json
from sklearn.metrics import average_precision_score
import csv
import os

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def normalize_multiple_coordinates(coord_str, width, height):
    matches = re.findall(r"\[(.*?)\]", coord_str)
    pixel_coords_list = []
    for match in matches:
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", match)
        if len(coords) != 4:
            continue
        x1 = float(coords[0])
        y1 = float(coords[1])
        x2 = float(coords[2])
        y2 = float(coords[3])
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:
            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height
        pixel_coords_list.append([x1, y1, x2, y2])
    return pixel_coords_list

def normalize_bounding_boxes(boxs):
    if boxs == [[]]:
        return []
    for i, box in enumerate(boxs):
        x1, y1, width, height = box
        x2 = x1 + width
        y2 = y1 + height
        boxs[i] = [x1, y1, x2, y2]
    return boxs

def find_and_remove_best_iou(target_box, box_list):
    best_index = -1
    best_iou = 0
    for i, box in enumerate(box_list):
        iou = calculate_iou(target_box, box)
        if iou > best_iou:
            best_iou = iou
            best_index = i
    if best_iou >= 0.5 and best_index != -1:
        del box_list[best_index]
    return box_list, 1 if best_iou >= 0.5 else 0

def eval_model(file_path, id, sign):
    datas = []
    with open(file_path, 'r') as file:
        for line in file:
            datas.append(json.loads(line))

    Pass = False
    if len(id) != 0:
        Pass = True
    
    test = []
    i = 1
    for data in datas:
        coord_str = data['predicted_answer']
        labeled_answer = data['labeled_answer']
        width = data['width']
        height = data['height']

        predicted_answer = normalize_multiple_coordinates(coord_str, width, height)
        labeled_answer = normalize_bounding_boxes(labeled_answer)

        if labeled_answer == []:
            if not Pass:
                id.append(i)
                sign.append(-1)
            test.append(len(predicted_answer))
            i += 1
            continue

        for j in range(len(labeled_answer)):
            if not Pass:
                id.append(i)
                sign.append(j+1)
        
        for label in labeled_answer:
            predicted_answer, x = find_and_remove_best_iou(label, predicted_answer)
            if x == 1:
                test.append(1)
            else:
                test.append(0)

        if not Pass:
            id.append(i)
            sign.append(-1)
        
        test.append(len(predicted_answer))
        i += 1


    return id, sign, test

def extract_model_name(filename):
    match = re.search(r'VQDv1_(.*)\.jsonl', filename)
    if match:
        return match.group(1)
    return None

def apply_eval_model_to_directory(directory, id, sign, test_set):
    for filename in directory:  
        model_name = extract_model_name(filename)
        id, sign, test = eval_model(filename, id, sign)
        test_set.append((model_name, test)) 
    return id, sign, test_set

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple file paths.")

    # Add the --path argument that accepts multiple values
    parser.add_argument('--path', nargs='+', help='List of file paths to process')

    # Parse the arguments
    args = parser.parse_args()

    # Access the list of file paths
    file_paths = args.path

    id, sign, test_set = apply_eval_model_to_directory(file_paths, [], [], [])
    for a in test_set:
        modelname, test = a
        print(f"Model name: {modelname}, length={len(test)}")

    # Debugging statements
    print(f"Total length of id: {len(id)}")
    print(f"Total length of sign: {len(sign)}")
    for model_name, test in test_set:
        print(f"Length of test for model {model_name}: {len(test)}")

    data = [{'question id': qid, 'sign': sg} for qid, sg in zip(id, sign)]

    # Add test results for each model to the data
    for model_name, test in test_set:
        for i, entry in enumerate(data):
            entry[model_name] = test[i]

    filename = 'hypo_testing/vqdv1.csv'

    # Get the fieldnames from the first dictionary in the data
    fieldnames = data[0].keys()

    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write the data rows
        for row in data:
            writer.writerow(row)

    print(f"CSV file '{filename}' written successfully.")
