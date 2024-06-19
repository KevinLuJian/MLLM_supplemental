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

    DVQA_dataset = load_dataset("dvqa_dataset")


    ans_file = open(args.answers_file, "a")

    i = 0
    import time
    start_time = time.time()
    for line in DVQA_dataset['val']:
        print(line.keys())
        questions = line["question"]

        prompt ='''please answer in one word.'''

        image_id = line['image_id']
        image = line['image']
        base64_image = encode_image(image)
        print(image)
        


        while True:
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                    {
                        "role": "user",
                        "content": [
                        {"type": "text", "text": f"{questions}, {prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        ],
                    }
                    ],
                max_tokens=700,
                )
                break
            except Exception as e:
                print(f"error: {e}, retrying...")
                continue


        predicted_answer = response.choices[0].message.content
        print(predicted_answer)
        labeled_answer = line["answer"]
        template_id = line['template_id']

        record = {"question_id": i,
                  "question": questions,
                  "predicted_answer": predicted_answer,
                  "labeled_answer": labeled_answer,
                  "image_id": image_id,
                  "template_id": template_id,
                  }
        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"question{i}: question:{questions}, predicted_answer: {predicted_answer}, labeled_answer: {labeled_answer}")
        
        i += 1

    ans_file.close()
    end_time = time.time()
    running_time = end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt-4o") # gpt-4-turbo
    parser.add_argument("--answers-file", type=str, default="/Users/jianlu/Desktop/Testing_result/GPT4V/DVQAa/DVQA_GPT4.json")
    args = parser.parse_args()

    args.answers_file = f"DVQA_{args.model}.jsonl"
    print(f"store the result in {args.answers_file}")
    eval_model(args)
