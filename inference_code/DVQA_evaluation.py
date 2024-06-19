import torch
from PIL import Image, ImageOps
import re
import json
from lavis.datasets.builders import load_dataset
import time
import argparse

# Load processor and model


parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
args = parser.parse_args()

if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'InstructBlip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl",torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
elif args.model == 'BLIP2':
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl",torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
    model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b",torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-13b")
    model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-13b",torch_dtype=torch.float16, low_cpu_mem_usage=True)

model.to("cuda:0") # We use Nividia A100 GPU for inference, so we use cuda:0.


# Load dataset and prepare output file
dvqa_dataset = load_dataset("dvqa_dataset")
ans_file = open('/path/to/answer_file.jsonl', "a")


# The main prompt is consistent accross all models, except some special token required by specific models, like LlavaNext requires
# prompt in the format [INST] <image> /INST], etc. We use the prompt example shown in HuggingFace. If the model didn't specify the 
# format it required, we use the default prompt.
def get_prompt(question, prompt):
    if args.model == 'LlavaNext':
        return f"[INST] <image>\n{question}, {prompt}/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        return f"USER: <image>\n {question}, {prompt} \nASSISTANT"
    else:
        return f"{question}, {prompt}"

i = 1
start_time = time.time()
for a in dvqa_dataset['val']:
    image = a['image']
    question = a['question']
    labeled_answer = a['answer']
    template_id = a['template_id']
    question_id = a['question_id']
    image_name = a['image_path']

    prompts = "please answer in one word" # Default prompt
    prompts = get_prompt(question,prompts)
    
    inputs = processor(images=image, text=prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=50,
            top_p=0.9,
            repetition_penalty=1.2,
            length_penalty=1.5,
            temperature=0,
        )
    

    answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
    record = {
        "question_index": i,
        "question": question,
        "question_id": question_id,
        "image_id": image_name,
        "predicted_answer": answer,
        "labeled_answer": labeled_answer,
        "template_id": template_id,
        "model_id": args.model,
    }

    ans_file.write(json.dumps(record) + "\n")
    ans_file.flush()
    print(f"{i} questions have been processed")
    i += 1


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()
