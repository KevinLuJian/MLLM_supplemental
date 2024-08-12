
import torch
from PIL import Image, ImageOps
import json
from lavis.datasets.builders import load_dataset
import time
import argparse
import re
# Load processor and model


parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--answer', type=str, required=True)
args = parser.parse_args()

if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'InstructBlip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
elif args.model == 'BLIP2':
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, low_cpu_mem_usage=True).to(0)
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)

model.to(args.device)

# Load dataset and prepare output file
tallyqa_dataset = load_dataset("tallyqa_dataset")
ans_file = open(args.answer, "a")


def get_prompt(question, prompt):
    if args.model == 'LlavaNext':
        return f"[INST] <image>\n{question}, {prompt}/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        return f"USER: <image>\n {question}, {prompt} \nASSISTANT:"
    else:
        return f"{question}, {prompt}"

def extract_answer(answer):
    if args.model == 'LlavaNext':
        match = re.search(r"INST\](.*)$", answer)
        # Check if the match was found
        if match:
            extracted_text = match.group(1).strip()
            # print("Extracted text:", extracted_text)
            return extracted_text
        else:
            return None
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        match = re.search(r'ASSISTANT:(.*)', answer, flags=re.DOTALL)
        return match.group(1).strip()
    else:
        return answer
start_time = time.time()

for i, a in enumerate(tallyqa_dataset['test'], start=1):
    image = a['image']
    question = a['question']
    labeled_answer = a['answer']
    issimple = a['issimple']
    image_name = a['image_id']

    
    prompt = "please answer in one word"
    prompt = get_prompt(question, prompt)


    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True).to(args.device)

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
    answer = extract_answer(answer)
   
    record = {
        "question_index": i,
        "question": question,
        "image_id": image_name,
        "predicted_answer": answer,
        "labeled_answer": labeled_answer,
        "issimple": issimple,
        "model_id": args.model,
    }
    ans_file.write(json.dumps(record) + "\n")
    ans_file.flush()
    print(f"{i} questions have been processed, answer = {answer}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()
