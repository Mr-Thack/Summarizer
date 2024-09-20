#!/usr/bin/python

import sys
import ollama
import pandas
import json

model="llama3.1:8b-instruct-fp16"

def prompt(system_prompt, question):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': question
        }
    ])
    return response['message']['content']

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

if __name__ == "__main__":
    filebase = None
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
        
    # if "blurb" in sys.argv:
    #    commonalities = [c[0] for c in pandas.read_json(filebase + "_commonalities.json").values]
    #    blurbs = []
    #    print(commonalities, "\n")
    #    res = prompt(PROMPTS["BLURB"], commonalities)
    #    print(res, "\n")
    #    write(res, filebase + "_blurbs.json")

