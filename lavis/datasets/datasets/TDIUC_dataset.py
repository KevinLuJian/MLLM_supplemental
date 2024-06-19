from collections import OrderedDict
import json
import os
import torch
import random
import re
from PIL import Image
import os

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset
'''
[{"question": "Where is the broccoli in the image?", 
"gtbox": [[250, 229, 316, 245]], this is the bounded box, that point to the answer
"question_type": "simple", 
"question_id": 0, 
"image_id": 9, 
"width": 640, 
"height": 480, 
"file_name": "COCO_train2014_000000000009.jpg", 
"split": "train"}
'''
def replace_image_id(filename, new_id):
    # Use regex to split the filename at the last occurrence of digits before the extension
    base, old_id, extension = re.match(r"^(.*?)(\d+)(\.\w+)$", filename).groups()
    
    # Convert new_id to string
    new_id_str = str(new_id)
    
    # Calculate how many digits were in the original ID and pad the new ID accordingly
    needed_length = len(old_id)
    new_id_str_padded = new_id_str.zfill(needed_length)
    
    # Combine the parts into the new filename
    new_filename = f"{base}{new_id_str_padded}{extension}"
    return new_filename


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "question": ann["question"],
                "question_type": ann["question_type"],
                "question_id": ann["question_id"],
                "answer": ann["labeled_answer"],
                "image_id": ann["image_id"],
            }
        )


class TDIUCDataset(BaseDataset, __DisplMixin):
    # vis_processor: responsible for processing images
    # text_processor: responsible for processing text(questions)
    # vis_root: where images are stored
    # ann_paths: Paths to annotation files containing metadata(data provides info about other data)
    # about the images, questions, and answers.
    # this is the constructor
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    # this method allows the dataset class to be indexed using square brackets.
    def __getitem__(self, index):
        # It retrieves the annotation (metadata including the question and answer details) for the given index.
        ann = self.annotation[index]

        # Constructs the full path to the image file.
        image_name = replace_image_id("COCO_val2014_000000000294.jpg", ann["image_id"])
        image_path = os.path.join(self.vis_root, image_name)
        image = None
        if os.path.exists(image_path): 
            # Opens the image file, converting it to RGB format to ensure consistency in image mode across the dataset.
            image = Image.open(image_path).convert("RGB")
            # process the image using vis_processor
            image = self.vis_processor(image)
            # process the questions(text) by text_processor

        question = self.text_processor(ann["question"])
        question_id = self.text_processor(ann["question_id"])
        answer = self.text_processor(ann["labeled_answer"])
        # answer_list = self.text_processor(ann["answer_list"])
        question_type = self.text_processor(ann["question_type"])

        return {
            "image_name": image_name,
            "image": image,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "question_type": question_type,
        }

class TDIUCInstructDataset(TDIUCDataset, __DisplMixin):
    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data

TDIUCEvalDataset = TDIUCDataset