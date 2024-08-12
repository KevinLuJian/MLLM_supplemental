## Introduction
This is the source code to reproduce the evaluations in "Revisiting Multi-Model LLM Evaluations". We provides the datasets, evaluation scripts, and example answer files for researchers to reproduce our results.

## Download the necessary datasets
We provide datasets auto-downloader based on Lavis library, to run the auto-downloader, first install Lavis, then run:
python downloading
This will download all the necessary images and datasets.

## Run inference on different models
The inference script will generate answer_file, which include both predicted answer and labeled answer, formatted as jsonl file.
To run inference, please first install LAVIS library, following the instructions(https://github.com/salesforce/LAVIS). After that, please download the LAVIS library we folk and cover it to the original one [download LAVIS(modified)](https://github.com/KevinLuJian/MLLM_supplemental/raw/main/lavis)


For BLIP2, InstructBLIP, LLaVA-1.5(7b), LLaVA-1.5(13b),LLaVA-NeXT(7b),we leverage the Huggingface Library, loading these model directly, and perform inference. [Example inference codes](https://github.com/KevinLuJian/MLLM_supplemental/raw/main/inference_code)

For GPT-4v/GPT-4o, we use the API from Open-AI to perform inference.[Example inference codes](https://github.com/KevinLuJian/MLLM_supplemental/raw/main/GPT-inference)

## Evaluation Script
Once you have the jsonl file, you can evaluate the performance with evaluation script.
To evaluate the performance of the models on the datasets, we provide evaluation scripts for each dataset. Please prepare the answer files in the format of the question-answer pairs we provided. You can download the evaluation scripts from this repository:

- [Download evaluation scripts](https://github.com/KevinLuJian/MLLM_supplemental/tree/main/eval_script)

## Example Answer Files
For each dataset, we provide example answer files in the format of the question-answer pairs, generated by our experiments. The example answer files can be found at:

- [Download example answer files](https://github.com/KevinLuJian/MLLM_supplemental/tree/main/Evaluation_result(Ours))

The most important components are the predicted answer and the labeled answer.

## Data Sampling Script
For the datasets VQDv1, TDIUC, and DVQA, where we sample a portion of the original testing datasets, we provide the sampling scripts that show how the datasets are being sampled, with detailed instructions in the readme file.

- [Download data sampling scripts](https://github.com/KevinLuJian/MLLM_supplemental/tree/main/datasets_sampling)

## License
&copy; 2024 Multi-Model LLM Evaluations. All rights reserved.
