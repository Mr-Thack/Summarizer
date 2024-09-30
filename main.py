#!/usr/bin/python

import sys
import pandas
import json
import argparse

# This section is for flags
DEBUG=None


MODEL = None

LOCAL_MODEL_LIGHTNING="llama3.2:1b-instruct-q4_0"
LOCAL_MODEL_FAST="llama3.1:8b-instruct-q8_0"
LOCAL_MODEL_SMART="llama3.1:8b-instruct-fp16"
CLOUD_MODEL_FAST="gpt4o-mini"
CLOUD_MODEL_SMART="gpt4o"


client = None
def prompt(system_prompt, question):
    if 'ollama' in sys.modules:
        response = ollama.chat(model=MODEL, messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
            ], options=dict(num_token=1024, num_thread=6))
        return response['message']['content']
    else:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )
        retrun completion.choices[0].message

def write(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

PROMPTS = {
    "SUMMARIZE": "Your job is to summarize, as concisely and accurately as possible, the Enhancement Requests for CTLS. Enhancement Requests are suggestions for improving CTLS (AKA CTLS Learn), a Learning Management System used in Cobb County. No headers or title are needed. Summarize the following requests as concisely and accurately as possible.",
    "SORT": "Your job is to sort each request for improving CTLS, a Learning Management System used in Cobb County, by finding commonalities bewteen requests. So, basically, you need to find requests that are repeats or similar. Pick category names which are descriptive of the specific issue or suggestion described. They should be descriptive enough that someone who has not read the requests should be able to roughly understand what the category refers to. If you do not think a request fits into the categories on this list, then you may suggest a new category by simply responding with the name of the new category. Again, JUST GIVE THE NAME OF THE NEW CATEGORY. No explanation or introduction is needed. Be accurate and descriptive. ONLY SUGGEST THE NAME OF 1 CATEGORY THAT BEST FITS THE REQUEST. DO NOT SHARE ANY REASONING, ONLY RESPOND WITH THE NAME OF THE SELECTED CATEGORY. Here is the list so far: ",
    "BLURB": "Your job is to find what actions would provide the most benefit for CTLS, the Learning Management System used in Cobb County. Provided to you are a list of teacher requests. Determine what actions would most quickly improve CTLS."
}

def prompt_sort(curlist):
    return PROMPTS["SORT"] + ", ".join(curlist)

def print_list(l):
    print("\n\n\n[\n")
    for i in l:
        print("\t", i, "\n")
    print("]\n\n\n")

def dpr(*args):
    if DEBUG:
        print(*args)

import pandas as pd
import os

def convert(excel_file_path):
    """
    Converts all sheets in an Excel file to separate CSV files.
    
    Parameters:
    - excel_file_path: str, path to the Excel file
    - output_dir: str, directory where the CSV files should be saved
    
    Each CSV file will have the same name as the sheet in the Excel file.
    """
    
    output_dir = excel_file_path.split(".")[0]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load all sheets into a dictionary
    excel_data = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name=None)  # sheet_name=None reads all sheets
    
    # Iterate over each sheet and save as CSV
    for sheet_name, data in excel_data.items():
        # Define the output CSV file path
        csv_file_path = os.path.join(output_dir, f"{sheet_name}.csv")
        print(data, "\n") 
        # Save each sheet's data to a CSV file
        print(csv_file_path)
        data.to_csv(csv_file_path, index=None)
        dpr(f"Saved sheet '{sheet_name}' to {csv_file_path}")

# Example usage:
# excel_sheets_to_csv("example.xlsx", "output_directory")

def summarize_file(f):
    fdata = pandas.read_csv("data/" + f + ".csv").to_dict('records')
    summaries = []
    for data in fdata:
        data = json.dumps(data)
        dpr(data)
        res = prompt(PROMPTS["SUMMARIZE"], data)
        dpr(res, "\n")
        summaries.append(res)
        write(summaries, "data/" + f + ".json")
    return summaries



def summarize(target):
    target_list = [target]
    if target == "ALL":
        target_list = [f.replace(".csv", "") for f in os.listdir("data")]
    sums = []
    for t in target_list:
        sums.append(summarize_file(t))
    return sums

def sort(target):
    dpr(target)

def blurb(target):
    dpr(target)


if __name__ == "__main__":
    filebase = None

    parser = argparse.ArgumentParser(
                            prog="CTLS Enhancement Request Summarizer",
                            description="Uses AI to summarize Enhancement Requests for CTLS",
                            epilog="Email AbdulMuqeet.Mohammed@yahoo.com for help")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Define the targets and actions that require a target (students/teachers)
    actions = ('convert', 'sort', 'summarize', 'blurb')
    actions_fns = (convert, sort, summarize, blurb)

    for action in actions:
        subparser = subparsers.add_parser(action, help=f'{action.capitalize()} target')
        subparser.add_argument('target', default="all", help='Target (sheet) to operate on')


    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output')
  
    parser.add_argument('--locallightning', action='store_true', help='Optional local processing 4 bit')
    parser.add_argument('--localfast', action='store_true', help='Optional local processing 8 bit')
    parser.add_argument('--localsmart', action='store_true', help='Optional local processing 16 bit')

    args = parser.parse_args()

    if args.locallightning or args.localfast or args.localsmart:
        import ollama
        MODEL = LOCAL_MODEL_SMART if args.localsmart else LOCAL_MODEL_FAST if args.localfast else LOCAL_MODEL_LIGHTNING if args.locallightning else LOCAL_MODEL_SMART 
    else:
        import OpenAI, Model
        client = OpenAI()
        MODEL = CLOUD_MODEL_FAST

    DEBUG = args.debug
   
    fn = actions_fns[actions.index(args.command)]
    fn(args.target)

    """
    if "teachers" in sys.argv:
        filebase = "teachers"
    elif "students" in sys.argv:
        filebase = "students"
    elif "convert" not in sys.argv:
        print("ERROR: filebase needed! Teachers or Students")
        exit

    if "convert" not in sys.argv and "summarize" not in sys.argv and "sort" not in sys.argv and "blurb" not in sys.argv:
        print("ERROR: instruction needed! summarize, sort, or blurb")
        exit

    if "convert" in sys.argv:
        fdata = pandas.read_excel("data.xlsx")
        print(fdata)

    if "summarize" in sys.argv:
        fdata = pandas.read_csv(filebase + ".csv").to_dict('records')
        summaries = []
        for data in fdata:
            data = json.dumps(data)
            print(data)
            res = prompt(PROMPTS["SUMMARIZE"], data)
            print(res, "\n")
            summaries.append(res)
            write(summaries, filebase + ".json")
        print_list(summaries)
    
    if "sort" in sys.argv:
        summaries = pandas.read_json(filebase + ".json").values
        commonalities = []
        for summary in summaries:
            summary = summary[0]
            PROMPT = prompt_sort(commonalities) # Generates Prompt
            print(summary)
            res = prompt(PROMPT, summary)
            if res not in commonalities:
                commonalities.append(res)
            print(res, "\n")
            write(commonalities, filebase  + "_commonalities.json")
        print_list(commonalities)

    if "blurb" in sys.argv:
        summaries = [c[0] for c in pandas.read_json(filebase + ".json").values]
        
        res = prompt(PROMPTS['BLURB'], '\n'.join(summaries))
        print(res, "\n")
        write(res, filebase + "_blurb.txt")
        
    if "blurb" in sys.argv:
       commonalities = [c[0] for c in pandas.read_json(filebase + "_commonalities.json").values]
       blurbs = []
       print(commonalities, "\n")
       res = prompt(PROMPTS["BLURB"], commonalities)
       print(res, "\n")
       write(res, filebase + "_blurbs.json")
    """


