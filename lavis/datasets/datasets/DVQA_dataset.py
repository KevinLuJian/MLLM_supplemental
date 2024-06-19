"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import torch
import random

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset

"""
# {'question': 'What is the label of the second bar from the bottom in each group?', 
'question_id': 120000013, 
'template_id': 'data', 
'answer': 'bond', 
'image': 'bar_train_00200000.png', 
'answer_bbox': [295.4375, 379.8534658136745, 54.5, 35.0]}"""

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "image_id": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "template_id": ann["template_id"],
                "answer": ann["answer"],
                "image": sample["image"],
                "answer_bbox": ann["answer_bbox"],

            }
        )


class DVQADataset(BaseDataset, __DisplMixin):
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
        #It retrieves the annotation (metadata including the question and answer details) for the given index.
        ann = self.annotation[index]
        # print(ann.keys())
        # Constructs the full path to the image file.
        image_path = os.path.join(self.vis_root, 'images')
        image_path = os.path.join(image_path, ann["image"])
        # Opens the image file, converting it to RGB format to ensure consistency in image mode across the dataset.
        image = Image.open(image_path).convert("RGB")
        # process the image using vis_processor
        image = self.vis_processor(image)
        # process the questions(text) by text_processor
        question = self.text_processor(ann["question"])

        answers = self.text_processor(ann["answer"])
        image_id = self.text_processor(ann["image"])
        
        template_id = self.text_processor(ann["template_id"])

        return {
            "image_id": image_id,
            "image": image,
            "question": question,
            "answer": answers,
            "template_id": template_id,
        }


class DVQAInstructDataset(DVQADataset, __DisplMixin):
    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data

DVQAEvalDataset = DVQADataset
# class DVQAEvalDataset(BaseDataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)