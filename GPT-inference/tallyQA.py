from openai import OpenAI
import os
import json
import argparse
import base64
from lavis.datasets.builders import load_dataset
from io import BytesIO

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def eval_model(args):  
    client = OpenAI()

    ans_file = open(args.answers_file, "a")

    tallyQA = load_dataset("tallyqa_dataset")
    i = 0
    import time
    start_time = time.time()
    for line in tallyQA['test']:
        question = line["question"]
        print(line.keys())
        prompt = "please answer in one word."

        image_id = line['image_id']
        image = line['image']
        base64_image = encode_image(image)

        while True:
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{question}, {prompt}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )
                break
            except Exception as e:
                print(f"error: {e}, retrying...")
                continue
            

        predicted_answer = response.choices[0].message.content
        labeled_answer = line["answer"]
        issimple = line['issimple']

        record = {
            "question id": i,
            "image": image_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "labeled_answer": labeled_answer,
            "issimple": issimple,
        }
        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"question{i}: question:{question}, predicted_answer: {predicted_answer}, labeled_answer: {labeled_answer}")
        
        i += 1

    ans_file.close()
    end_time = time.time()
    running_time = end_time - start_time
    print(f"running time: {running_time}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o") #gpt-4-turbo
    parser.add_argument("--answers-file", type=str, default="GPT4V/TallyQA/TallyQA_4v_10000.json")
    args = parser.parse_args()
    args.answers_file = f"TallyQA_{args.model}.jsonl"
    print(f"store the result in {args.answers_file}")
    eval_model(args)
