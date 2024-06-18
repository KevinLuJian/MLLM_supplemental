## Introduction
This is the source code to reproduce the evaluations in "Revisiting Multi-Model LLM Evaluations". We provides the datasets, evaluation scripts, and example answer files for researchers to reproduce our results.

## Datasets

### VQDv1
- [Download VQDv1 question-answer pairs](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/VQDv1_sampling.json)
- [Download VQDv1 images (val2014)](http://images.cocodataset.org/zips/val2014.zip)

### TallyQA
- [Download TallyQA testing dataset](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TallyQA_test.json)
- [Download TallyQA images part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
- [Download TallyQA images part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### TDIUC
- [Download TDIUC question-answer pairs](https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TDIUC_sampling.json)
- [Download TDIUC images](https://drive.google.com/file/d/1Hevf7eQNzg-qlXbfz9nPbATmQciexkDp/view?usp=share_link)

### DVQA
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
