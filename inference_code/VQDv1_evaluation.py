from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from PIL import Image, ImageOps
import re
import json
import time
import argparse
import os

# Load processor and model
parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
args = parser.parse_args()


if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, low_cpu_mem_usage=True).to(0)
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")


ans_file = open('/path/to/answer.jsonl', 'a')



def get_prompt(question):
    if args.model == 'LlavaNext':
        prompt = 'Please generate a list of bounding boxes coordinates of the region this query describes. Use the format [[x\_min,y\_min,x\_max,y\_max]....]. Do not respond in sentences, and only generate the bounding boxes. Respond with an empty list [[]], if no such region exists in the image. '
        return f"[INST] <image>Query: \n{question}, prompt: {prompt}[/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        prompt = 'Please answer the query by generating a list of bounding box coordinates around the objects the query is asking, and if no such object exists in the image, answer: [[]]'
        return f"USER: <image>\Query: {question},Instruction: {prompt}\nASSISTANT:"


def extract_non_inst_text(text):
    non_inst_text = ''
    if args.model == 'LlavaNext':
        non_inst_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        match = re.search(r'ASSISTANT:(.*)', text, flags=re.DOTALL)
        if match:
            non_inst_text = match.group(1).strip()
    return non_inst_text

i = 0
start_time = time.time()

from lavis.datasets.builders import load_dataset
dataset = load_dataset("vqdv1_dataset")
    
for a in dataset:

    image = a['image']
    question = a['question']
    labeled_answer = a['answer']
    question_type = a['question_type']
    question_id = a['question_id']
    width = a['width']
    height = a['height']
    image_name = a['image_path']
    prompt = get_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=2,
            max_new_tokens=512,
            top_p=0.9,
            repetition_penalty=1.2,
            length_penalty=1,
            temperature=0,
        )
    answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
    answer = extract_non_inst_text(answer)

    record = {
        "question_index": i,
        "question": question,
        "question_id": question_id,
        "image_id": image_name,
        "predicted_answer": answer,
        "labeled_answer": labeled_answer,
        "question_type": question_type,
        "model_id": args.model,
        "width":width,
        "height":height,
        "prompt": prompt
    }
    ans_file.write(json.dumps(record) + "\n")
    ans_file.flush()
    print(f"{args.model}, {i} questions have been processed, question:{question}, predicted_answer:{answer}, label: {labeled_answer}")
    i += 1

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()

