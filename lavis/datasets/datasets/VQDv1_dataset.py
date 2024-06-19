from collections import OrderedDict
import json
import os
import torch
import random

from PIL import Image

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

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "question": ann["question"],
                "question_type": ann["question_type"],
                "question_id": ann["question_id"],
                "file_name": ann["file_name"],
                "answer_gtbox": ann["gtbox"],
                "split": ann["split"],
                "question_type": ann["question_type"],
            }
        )


class VQDv1Dataset(BaseDataset, __DisplMixin):
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
        # print(ann.keys())
        # Constructs the full path to the image file.
        image_path = os.path.join(self.vis_root, 'val2014')
        image_path = os.path.join(image_path, ann["file_name"])
        # Opens the image file, converting it to RGB format to ensure consistency in image mode across the dataset.
        image = Image.open(image_path).convert("RGB")
        # process the image using vis_processor
        image = self.vis_processor(image)
        # process the questions(text) by text_processor

        question = self.text_processor(ann["question"])

        question_id = self.text_processor(ann["question_id"])
        answer_gtbox = self.text_processor(ann["gtbox"])
        image_name = self.text_processor(ann["file_name"])
        width = self.text_processor(ann["width"])
        height = self.text_processor(ann["height"])
        question_type = self.text_processor(ann["question_type"])

        return {
            "question_id": question_id,
            "image": image,
            "image_path": ann['file_name'],
            "question": question,
            "answer": answer_gtbox,
            "question_type": question_type,
            "width": width,
            "height": height
        }

class VQDv1InstructDataset(VQDv1Dataset, __DisplMixin):
    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data

VQDv1EvalDataset = VQDv1Dataset