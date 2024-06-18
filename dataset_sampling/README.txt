This folder, contains the datasets the we used by sampling the orginal datasets, we have incorporate them in LAVIS Library,
and it can be accessed directly via LAVIS with ease. We also include the code that we used to achieve the sampling process.
If you prefer to verify the sampling process, we also include the code that we used to achieve the sampling process.
Below is how each scripts are used.

DVQA: 
1. First, please download the orginal datasets QA pairs: 
https://drive.google.com/file/d/1VKYd3kaiCFziSsSv4SgQJ2T5m7jxuh5u/view?usp=sharing
2. Please set: DVQA_easy = '/path/to/val_easy_qa.json'
               DVQA_hard = '/path/to/val_hard_qa.json' in line 6 and 7, DVQA_sampling.py.
3. Run: python datasets_sampling/script/DVQA_sampling.py --target_path path/to/new/DVQA/json. 

TallyQA:
1. We use the entire set of TallyQA, which can be downloaded via https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true, 
2. the test.json is the test set we used in this project

VQDv1:
1. First, please download the orignal datasets QA pairs https://github.com/manoja328/VQD_dataset/blob/master/vqd_dataset.zip
2. Please set: original_val='/path/to/VQDv1/val.json' in line 7 /script/VQDv1_sampling.py
3. Run: python datasets_sampling/script/VQDv1_sampling.py --target_path path/to/new/VQDv1/json. 
 
TDIUC:
1. First, Please download the orignal datasets QA pairs, notice that TDIUC has two separate json file for validation set, one for questions, one for annotations. 
https://kushalkafle.com/data/TDIUC.zip

2. Please set: TDIUC_question = '/PATH/TO/OpenEnded_mscoco_val2014_questions.json'
               TDIUC_annotation = '/PATH/TP/mscoco_val2014_annotations.json' in line 6, 7.
3. Run: python datasets_sampling/script/TDIUC_sampling.py --target_path /path/to/target/TDIUC/json.

Please notice that the sampling process is random. While our sampling approach ensure answer variety, and maintain a constant porportion of every question type,
everytime you create the subset, the resulting datasets can be different.
    
