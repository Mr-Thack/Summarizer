#!/usr/bin/python

import sys
import json
import argparse
import yaml
import pandas as pd
import os
from pydantic import BaseModel
from dataclasses import dataclass

# This section is for flags
DEBUG = None
def select_model(name):
    global MODEL
    MODEL = name
    global CLIENT
    # Ok, this isn't foolproof, but like, it works for now
    if "gpt" not in name:
        from ollama import Client
        CLIENT = Client(host="http://localhost:11434")
    else:
        from openai import OpenAI, Model
        # Do not know why, but "global" fixes it
        CLIENT = OpenAI()

def select_model_from_nickname(nickname):
    if nickname in MODEL_NICKNAMES:
        model = MODEL_NICKNAMES[nickname]
        select_model(model)
        return True
    else:
        print("Nickname {} not found!".format(nickname))
        print("Crashing since we don't know what AI Model to use.")
        print("Select one of these model names using `--model MODEL_NAME`:")
        for alias, model in MODEL_NICKNAMES:
            print("\t{}: Uses {}".format(alias,model))
        return False

class DictObject(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    @classmethod
    def from_dict(cls, d):
        return json.loads(json.dumps(d), object_hook=DictObject)


def prompt(system_prompt, question, response_model=None):
    args = {
        'model': MODEL,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ]
    }
    if response_model and 'ollama' not in sys.modules:
        # args['temperature'] = 0
        args['response_format'] = response_model 
    response = None
    if 'ollama' in sys.modules:
        response = DictObject.from_dict(CLIENT.chat(**args))
    else:
        completion = None
        if response_model:
            completion = CLIENT.beta.chat.completions.parse
        else:
            completion = CLIENT.chat.completions.chat
        completion = completion(**args)
        response = completion.choices[0]
    if response_model:
        return response.message.parsed
    else:
        return response.message.content

def write(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_config():
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

PROMPTS = CONFIG['PROMPTS']
MODEL_NICKNAMES = CONFIG['MODEL_NICKNAMES']

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

class Topic(BaseModel):
    name: str
    description: str

def sort_req(summary, commonalities):
    dpr(summary)
    mprompt = prompt_sort(commonalities) # Generates Prompt
    res = prompt(mprompt, summary, Topic)
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
    #   Subject Category Name 1: {
    #       Categorization Name 1: {
    #           desc: Description,
    #           requests: [req ids]
    #       },
    #       ...
    #   },
    #   Subject Category Name 2: ...
    # }
    subjects = dict() # Holds requests corresponding to what topic to what category 
    for i, data in enumerate(SortReader(f)):
        # First, find get the index (The Subject Name)
        index = data['request']['Category']

        # dict.get() will return [] if that entry is null
        subject = subjects.get(index, {})
        dpr("Subject: {}".format(index))
        # Then get a category for the request
        topic: Topic = sort_req(data['summary'], subject)
        if topic is None:
            pass # Something went wrong, I don't care enough to debug 


        # if request_category in current request_categories of subject_category
        if topic.name in subject.keys():
            # Just append the current req id
            subject[topic.name]["requests"].append(i)
        else:
            # Make a list only containing the current req id
            subject[topic.name] = {
                "description": topic.description,
                "requests": [i]
            }

        # Then put the modified subject category back into the big list
        subjects[index] = subject
        write(subjects, get_filename(f, 'commonalities.json'))
    return subjects 

@targeter
def sort(targets):
    return [sort_file(t) for t in targets]

def read_sort_file(f):
    commonalities = dict()
    with open(get_filename(f + ".commonalities", 'json'), 'r') as fp:
        commonalities = json.load(fp)
    
    requests = read_csv(f)

    for subject in commonalities.keys():
        print("\n\n***Enhancement Category: {}***".format(subject))
        for category in commonalities[subject].keys():
            print("\n**Subject Category: {}**".format(category))
            print("*{}*".format(commonalities[subject][category]['description']))
            for request in commonalities[subject][category]['requests']:
                print(requests[request]['Description'])
                print(requests[request]['Additional Information'], '\n')


@targeter
def read_sort(targets):
    return [read_sort_file(t) for t in targets]

def blurb(target):
    dpr(target)


def main():
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

    parser.add_argument('--model', choices=MODEL_NICKNAMES.keys(), default='cloud_smart', help='Optionally select model (default: cloud_smart=gpt4o)')


    args = parser.parse_args()

    select_model_from_nickname(args.model)
    
    global DEBUG
    DEBUG = args.debug
   
    fn = actions_fns[actions.index(args.command)]
    fn(args.target)

    """
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

if __name__ == "__main__":
    main()
