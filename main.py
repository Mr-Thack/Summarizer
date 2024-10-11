#!/usr/bin/python

import sys
import json
import argparse
import yaml
import pandas as pd
import os

# This section is for flags
DEBUG = None
CLIENT = None

MODEL = None

LOCAL_MODEL_LIGHTNING="llama3.2:1b-instruct-q2_K"
LOCAL_MODEL_FAST="llama3.1:8b-instruct-q8_0"
LOCAL_MODEL_SMART="llama3.1:8b-instruct-fp16"
CLOUD_MODEL_FAST="gpt-4o-mini"
CLOUD_MODEL_SMART="gpt-4o"


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
        completion = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return completion.choices[0].message.content

def write(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_config():
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

PROMPTS = CONFIG['PROMPTS']

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
        dpr(data, "\n") 
        # Save each sheet's data to a CSV file
        dpr(csv_file_path)
        data.to_csv(csv_file_path, index=None)
        dpr(f"Saved sheet '{sheet_name}' to {csv_file_path}")


def summarize_req(r):
    data = json.dumps(r)
    dpr(data)
    res = prompt(PROMPTS["SUMMARIZE"], data)
    dpr(res, '\n')
    return res

def sort_req(summary, commonalities):
    dpr(summary)
    mprompt = prompt_sort(commonalities) # Generates Prompt
    res = prompt(mprompt, summary)
    dpr(res, "\n")
    return res

def get_filename(f, ext):
    return "data/{}.{}".format(f, ext)

class Reader:
    def __init__(self, filename: str):
        self.filename = filename

    # Get all of the data
    def __read__(self):
        self.data = []

    def __iter__(self):
        self.index = -1
        dpr("***{}***\n".format(self.filename))
        self.__read__()
        return self

    def __next__(self):
        self.index += 1
        if self.index == len(self.data):
            raise StopIteration
        dpr("{} of {}".format(self.index + 1, len(self.data)))
        return self.data[self.index]


def read_csv(f):
    return pd.read_csv(get_filename(f, 'csv')).to_dict('records')

def read_json(f):
    return [r[0] for r in pd.read_json(get_filename(f, "json")).values]

# Iterates through a CSV File
class CSVReader(Reader):
    def __read__(self):
        self.data = read_csv(self.filename)

class JSONReader(Reader):
    def __read__(self):
        # For some reason, it returns obj[0] insetead of obj for each, and I don't know why
        self.data = read_json(self.filename)

class SortReader(Reader):
    def __read__(self):
        self.data = [
            {
                'request': d[0],
                'summary': d[1]
            }
            for d in zip(read_csv(self.filename), read_json(self.filename))
        ]

def get_target_list(ts):
    # ts = "ALL" or base of filename
    target_list = [ts]
    if ts == "ALL":
        target_list = list(set([f.split(".")[0] for f in os.listdir("data")]))
    return target_list

# This wraps any function with the get_target_list function
# get_target_list takes either a list of file root or "ALL" and turns that into a list of filesnames
def targeter(fn):
    def wrap(t):
        fn(get_target_list(t))
    return wrap

def summarize_file(f):
    summaries = []
    for data in CSVReader(f):
        summaries.append(summarize_req(data))
        write(summaries, get_filename(f, 'json'))
    return summaries

@targeter
def summarize(targets):
    return [summarize_file(t) for t in targets]

def sort_file(f):
    # This will hold an object like this:
    # {
    #   Subject Category Name 1: {
    #       Categorization Name 1: [req ids],
    #       Categorization Name 2: [req ids],
    #   },
    #   Subject Category Name 2: ...
    # }
    commonalities = {}
    for i, data in enumerate(SortReader(f)):
        # First, find get the index (The Subject Name)
        index = data['request']['Category']

        # dict.get() will return [] if that entry is null
        subject = commonalities.get(index, {})
        dpr("Subject: {}".format(index))
        # Then get a category for the request
        category = sort_req(data['summary'], subject)
      
        # if request_category in current request_categories of subject_category
        if category in subject.keys():
            # Just append the current req id
            subject[category].append(i)
        else:
            # Make a list only containing the current req id
            subject[category] = [i]

        # Then put the modified subject category back into the big list
        commonalities[index] = subject
        write(commonalities, get_filename(f, 'commonalities.json'))
    return commonalities

@targeter
def sort(targets):
    return [sort_file(t) for t in targets]

def read_sort_file(f):
    commonalities = []
    with open(get_filename(f + ".commonalities", 'json'), 'r') as fp:
        commonalities = json.load(fp)
    
    requests = read_csv(f)

    for subject in commonalities.keys():
        print("\n\n***Enhancement Category: {}***\n".format(subject))
        
        for category in commonalities[subject].keys():
            print("\n**Subject Category: {}**".format(category))
            for request in commonalities[subject][category]:
                print(requests[request], "\n")


@targeter
def read_sort(targets):
    return [read_sort_file(t) for t in targets]

def blurb(target):
    dpr(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                            prog="CTLS Enhancement Request Summarizer",
                            description="Uses AI to summarize Enhancement Requests for CTLS",
                            epilog="Email AbdulMuqeet.Mohammed@yahoo.com for help")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Define the targets and actions that require a target (students/teachers)
    actions = ('convert', 'sort', 'read_sort', 'summarize', 'blurb')
    actions_fns = (convert, sort, read_sort, summarize, blurb)

    for action in actions:
        subparser = subparsers.add_parser(action, help=f'{action.capitalize()} target')
        subparser.add_argument('target', default="all", help='Target (sheet) to operate on')


    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output')
  
    parser.add_argument('--locallightning', action='store_true', help='Optional local processing 4 bit')
    parser.add_argument('--localfast', action='store_true', help='Optional local processing 8 bit')
    parser.add_argument('--localsmart', action='store_true', help='Optional local processing 16 bit')
    parser.add_argument('--cloudsmart', action='store_true', help='Optional GPT4o')
    parser.add_argument('--cloudfast', action='store_true', help='Optional GPT4o-mini')
    
    args = parser.parse_args()

    if args.locallightning or args.localfast or args.localsmart:
        import ollama
        MODEL = LOCAL_MODEL_SMART if args.localsmart else LOCAL_MODEL_FAST if args.localfast else LOCAL_MODEL_LIGHTNING if args.locallightning else LOCAL_MODEL_SMART 
    else:
        from openai import OpenAI, Model
        CLIENT = OpenAI()
        MODEL = CLOUD_MODEL_SMART if args.cloudsmart else CLOUD_MODEL_FAST
    
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


