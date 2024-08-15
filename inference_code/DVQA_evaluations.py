import torch
from PIL import Image, ImageOps
import re
import json
from lavis.datasets.builders import load_dataset
import time
import argparse
from tqdm import tqdm
# Load processor and model

DEFAULT_PROMPT =  "please answer in one word"

parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
parser.add_argument('--device', type=str, required=False, default='cuda')
parser.add_argument('--answer', type=str, required=False, default='evaluation_results/DVQA/dvqa_test.jsonl')
args = parser.parse_args()
processor, model = None, None
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
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, low_cpu_mem_usage=True).to(0)
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'QwenVL':
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", torch_dtype=torch.float16, trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


model.to(args.device) # We use Nividia A100 GPU for inference, so we use cuda:0.


# Load dataset and prepare output file
dvqa_dataset = load_dataset("dvqa_dataset")
ans_file = open(args.answer, "a")


# The main prompt is consistent accross all models, except some special token required by specific models, like LlavaNext requires
# prompt in the format [INST] <image> /INST], etc. We use the prompt example shown in HuggingFace. If the model didn't specify the 
# format it required, we use the default prompt.
def get_prompt(question, prompt):
    if args.model == 'LlavaNext':
        return f"[INST] <image>\n{question}, {prompt}[/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        return f"USER: <image>\n {question}, {prompt} \nASSISTANT:"
    else:
        return f"{question}, {prompt}"

def get_input(question, prompts, image, image_path=None):
    if args.model == 'QwenVL':
        prompts = get_prompt(question, prompts)
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': f"{prompts}"},
            ])
        return query
    else:
        prompts = get_prompt(question, prompts)
        inputs = processor(images=image, text=prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        return inputs

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

i = 1
start_time = time.time()

# Initialize tqdm with the length of your dataset
with tqdm(total=len(dvqa_dataset['val'])) as pbar:
    for a in dvqa_dataset['val']:
        image = a['image']
        question = a['question']
        labeled_answer = a['answer']
        template_id = a['template_id']
        image_name = a['image_id']
        image_path = f'LAVIS/cache/DVQA/images/images/{image_name}'
        prompts =  DEFAULT_PROMPT # Default prompt

        inputs = get_input(question, prompts, image, image_path)

        with torch.inference_mode():
            if args.model == 'QwenVL':
                outputs = model.chat(tokenizer, query=inputs, history=None,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=100, 
                    top_p=0.9,
                    repetition_penalty=1.2,
                    length_penalty=1.5,
                    temperature=0)
            else:
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=100,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    length_penalty=1.5,
                    temperature=0,
                )

        if args.model == 'QwenVL':
            answer = outputs[0]
        else:
            answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
            
        answer = extract_answer(answer)
        record = {
            "question_index": i,
            "question": question,
            "image_id": image_name,
            "predicted_answer": answer,
            "labeled_answer": labeled_answer,
            "template_id": template_id,
            "model_id": args.model,
        }

        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"predicted = {answer}, labeled = {labeled_answer}")
        i += 1

        # Update the progress bar
        pbar.update(1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()