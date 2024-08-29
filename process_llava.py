import json
import re
import argparse
import os

def process_jsonl_file(input_file):
    # Define the regex pattern to match everything after 'assistant\n'
    pattern = re.compile(r'assistant\n(.+)', re.DOTALL)
    
    # Generate the output file name by inserting '_processed' before '.jsonl'
    output_file = f"{os.path.splitext(input_file)[0]}_processed.jsonl"
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            # Extract and process the predicted_answer key using regex
            if "predicted_answer" in data:
                predicted_answer = data["predicted_answer"]
                match = pattern.search(predicted_answer)
                
                if match:
                    # Extract the content after 'assistant\n' and strip it
                    data["predicted_answer"] = match.group(1).strip()
            
            # Write the modified data to the output file
            outfile.write(json.dumps(data) + '\n')
    
    print(f"Processed file saved to: {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a JSONL file and modify the predicted_answer field.")
    parser.add_argument("input_file", help="The input JSONL file to be processed.")
    
    args = parser.parse_args()
    
    # Process the JSONL file
    process_jsonl_file(args.input_file)