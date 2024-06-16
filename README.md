# Revisiting Multi-Model LLM Evaluations

**Authors**: Jian Lu, Shikhar Srivastava, Junyu Chen, Robik Shrestha, Manoj Acharya, Kushal Kafle, Christopher Kanan

## Introduction
In this paper, we evaluate popular multi-modal large language models (MLLMs) on four datasets created by our lab: VQDv1, TallyQA, TDIUC, and DVQA. These datasets are designed to assess the performance of multi-modal LLMs on various tasks, including visual question answering, counting, and image understanding. 

VQDv1, TallyQA, and DVQA evaluations are not based on complete datasets but instead use representative samples. This effectively shortens the testing sets, allowing developers to expedite the evaluation process while maintaining the diversity of the original datasets. This website provides the datasets, evaluation scripts, and example answer files for researchers to reproduce our results.

## Datasets

### VQDv1
VQDv1 requires the model to produce multiple bounding boxes instead of localizing only one object, thereby testing general query detection skills. Unlike typical referring expression datasets, which assert that every query will correspond to only one bounding box, VQDv1 queries ask the model to generate an uncertain number of bounding boxes, from 0 to N, posing an additional challenge to the model.

- [Download VQDv1 question-answer pairs](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/VQDv1_sampling.json)
- [Download VQDv1 images (val2014)](http://images.cocodataset.org/zips/val2014.zip)

### TallyQA
<p align="center">
  <img src="path/to/your/introduction_image.png" alt="Introduction Image" width="400"/>
  <br>
  <em>How many people are there?</em>
</p>
TallyQA tests models' visual grounding through counting skills. In addition to simple counting questions that the model can handle well with straightforward object detection, TallyQA also incorporates complex counting questions that demand sophisticated reasoning capabilities, such as pose estimation (e.g., "How many dogs are sitting?") and positional reasoning (e.g., "How many dogs are in front of the white building?").

- [Download TallyQA testing dataset](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TallyQA_test.json)
- [Download TallyQA images part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
- [Download TallyQA images part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### TDIUC
TDIUC tests the models' versatility across 12 tasks, including object, attribute, and activity recognition, as well as overall scene understanding. The meaningful categories of question types permit fine-grain analysis of the models' abilities from different perspectives, allowing us to identify the specific strengths and weaknesses of each model.

- [Download TDIUC question-answer pairs](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TDIUC_sampling.json)
- [Download TDIUC images](https://drive.google.com/file/d/1Hevf7eQNzg-qlXbfz9nPbATmQciexkDp/view?usp=share_link)

### DVQA
DVQA requires the models to interpret and analyze visual data in chart form, testing their ability to perform OCR and handle unusual words found in charts. The charts are all synthetically generated images, which pose additional challenges different from natural images.

- [Download DVQA question-answer pairs](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/DVQA_sampling.json)
- [Download DVQA images](https://drive.google.com/file/d/1iOSjgbqnTiLpMFuuRa3kIs3E_RxGkKmX/view?usp=share_link)

## Evaluation Script
To evaluate the performance of the models on the datasets, we provide evaluation scripts for each dataset. Please prepare the answer files in the format of the question-answer pairs we provided. You can download the evaluation scripts from this repository:

- [Download evaluation scripts](https://github.com/KevinLuJian/MLLM-evaluation/tree/main/eval_script)

Detailed instructions on how to use the scripts are available in the repository.

## Example Answer Files
For each dataset, we provide example answer files in the format of the question-answer pairs we used. The example answer files can be found at:

- [Download example answer files](https://github.com/KevinLuJian/MLLM-evaluation/tree/main/Evaluation_result(Ours))

The most important components are the predicted answer and the labeled answer.

## Data Sampling Script
For the datasets VQDv1, TDIUC, and DVQA, where we sample a portion of the original testing datasets, we provide the sampling scripts that show how the datasets are being sampled, with detailed instructions in the readme file.

- [Download data sampling scripts](https://github.com/KevinLuJian/MLLM-evaluation/tree/main/datasets_sampling)

## License
&copy; 2024 Multi-Model LLM Evaluations. All rights reserved.
