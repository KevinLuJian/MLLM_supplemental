import re
import json
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
    - box1, box2: List of [x1, y1, x2, y2] where (x1, y1) is the top-left
                  corner and (x2, y2) is the bottom-right corner.
                  
    Returns:
    - IoU: Intersection over Union (IoU) value.
    """
    # Extract coordinates for both boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate the coordinates of the intersection rectangle
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    # Calculate the area of the intersection rectangle
    inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    
    # Calculate the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate the area of the union
    union_area = box1_area + box2_area - inter_area
    
    # Calculate the IoU
    iou = inter_area / union_area
    
    return iou

def extract_qwen2(input_string):
    # Regular expression to find all <box> tags and extract the numbers inside them
    pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
    
    # Find all matches in the input string
    matches = re.findall(pattern, input_string)
    
    # Process and store the extracted numbers
    boxes = []
    for match in matches:
        x1, y1, x2, y2 = match
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    print(f"the coordinate is {boxes}")
    return boxes

def normalize_multiple_coordinates(coord_str, width, height,model_name):
    if model_name == "qwen":
        return extract_qwen2(coord_str)
    # Regular expression to extract all bounding box coordinates
    matches = re.findall(r"\[(.*?)\]", coord_str)
    # Initialize an empty list to store the converted pixel coordinates
    pixel_coords_list = []
    
    # Iterate over each match
    for match in matches:
        # Extract the individual coordinates from the matched string
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", match)
        
        if len(coords) != 4:
            continue
        
        
        # Convert the extracted strings to floats
        x1 = float(coords[0])
        y1= float(coords[1])
        x2 = float(coords[2])
        y2 = float(coords[3])
        # if the model output coordinate in the range of [0, 1], convert it to pixel coordinates.
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:
            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height
        
        # Append the pixel coordinates to the list
        pixel_coords_list.append([x1, y1, x2, y2])
    
    return pixel_coords_list



# For the labeled bounding box
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
    
    # Check if the best IoU is greater than or equal to 50%
    if best_iou >= 0.5 and best_index != -1:
        # Remove the bounding box with the highest IoU
        del box_list[best_index]
    
    return box_list, 1 if best_iou >= 0.5 else 0

def eval_model(file_path, model):
    datas = []
    with open(file_path, 'r') as file:
        for line in file:
            datas.append(json.loads(line))
       
    i = 0
    final_score = []
    dict = {}  # Assuming dict is initialized
    
    
    for data in datas:
        TP, FP, FN = 0, 0, 0
        coord_str = data['predicted_answer']
        labeled_answer = data['labeled_answer']
        width = data['width']
        height = data['height']

        predicted_answer = normalize_multiple_coordinates(coord_str, width, height,model)
        labeled_answer = normalize_bounding_boxes(labeled_answer)
    
        if len(labeled_answer) not in dict:
            dict[len(labeled_answer)] = []
        if predicted_answer == [] and labeled_answer == []: # If both the predicted and labeled answers are empty, prediction is correct
            dict[0].append(1)
            final_score.append(1)
            i += 1
            continue
        elif predicted_answer == [] and labeled_answer != []: # If the predicted answer is empty, but the labeled answer is not
            dict[len(labeled_answer)].append(0)
            i += 1
            final_score.append(0)
            continue
        elif predicted_answer != [] and labeled_answer == []: # If the labeled answer is empty, but the predicted answer is not
            dict[0].append(0)
            final_score.append(0)
            i += 1
            continue
        
        # Iterate over each labeled bounding box, and find the best IoU match
        for label in labeled_answer:
            predicted_answer, x = find_and_remove_best_iou(label, predicted_answer)
            if x == 1:
                TP += 1
        
        if len(predicted_answer) > 0:
            FP = len(predicted_answer)
        if len(labeled_answer) > 0:
            FN = len(labeled_answer)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
       
        final_score.append(f1)
        if len(labeled_answer) not in dict:
            dict[len(labeled_answer)] = []
        
        dict[len(labeled_answer)].append(f1)
        i += 1

    p = 0
    above = 0
    length = 0
    for key in sorted(dict.keys()):  # Sort keys in increasing order
        value = dict[key]
        if key < 5:
            print(f"Number of bounding boxes: {key}, Accuracy: {sum(value) / len(value)}")
            p += sum(value) / len(value)
        else:
            above += sum(value)
            length += len(value)
    
    p += sum(value) / len(value)

    print(f"Number of bounding boxes above 4: {length}, Accuracy: {above / length}") 
    print(f"Micro Accuracy: {p / 6}")
    print(f"Macro Accuracy: {sum(final_score) / len(final_score)}")

import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add the arguments
    parser.add_argument('--path', type=str, help='answer_path')
    parser.add_argument('--model', type=str, help='model_name')
    
    # Parse the arguments
    args = parser.parse_args()
    print(f"====================VQDv1 Evaluations {args.path}====================\n")
    eval_model(args.path,args.model)
    