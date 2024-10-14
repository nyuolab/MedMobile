import re
import sys
import unicodedata
from openai import OpenAI
import os
from datasets import load_dataset
import random
import json
import requests
from vllm import LLM, SamplingParams

# Set openai key if using gpt4o as engine.
os.environ['OPENAI_API_KEY'] = "OPEN AI KEY HERE"

def return_parted_rows(df, part_ind, part_ind_list):
    # Extract chapter numbers from the 'chapters' column
    df['chapter_num'] = df['current_chapter'].apply(lambda x: int(re.search(r'\d+', x).group()))
    
    # Determine the start and end range for the given part_ind
    start = part_ind_list[part_ind]
    end = part_ind_list[part_ind + 1] - 1 if part_ind + 1 in part_ind_list else df['chapter_num'].max()

    # Filter the DataFrame to include only rows with chapters within the range
    df_filtered = df[(df['chapter_num'] >= start) & (df['chapter_num'] <= end)]
    
    # Drop the temporary 'chapter_num' column if not needed
    df_filtered = df_filtered.drop(columns=['chapter_num'])
    
    return df_filtered

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)

    
def format_examples(examples):
    formatted_examples = []
    for row in examples:
        example = f'## Question {row["question"]} \n ## Answer {row["answer"]}'
        formatted_examples.append(example)
    return "\n".join(formatted_examples)

def extract_samples(task, numShot, model_prompt):
    questions, answer_choices, correct_answers = task_load(task, 'train')
    example_indexes = random.sample(range(len(questions)), numShot)
    example_list = []
    for i in example_indexes:
        example_list.append(model_prompt.format(question=questions[i], choices=format_choices(answer_choices[i]), answer=correct_answers[i]))
    return example_list

def task_load(task, split):
    if task=="medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        questions = [ds[i]['question'] for i in range(len(ds))]
        answer_choices = [ds[i]['options'] for i in range(len(ds))]
        correct_answers = [ds[i]['answer_idx'] for i in range(len(ds))]
        return questions, answer_choices, correct_answers
    
    elif task=="medmcqa":
        if split == 'test':
            split = 'validation'
        ds = load_dataset("openlifescienceai/medmcqa", split=split)
        questions = [ds[i]['question'] for i in range(len(ds))]
        answer_choices = [{"A": ds[i]['opa'], "B": ds[i]['opb'], "C": ds[i]['opc'], "D": ds[i]['opd']} for i in range(len(ds))]
        correct_answers = [chr(ds[i]['cop']+65) for i in range(len(ds))]
        return questions, answer_choices, correct_answers
    
    elif task=="medbullets_op4":
        path = "ADD MEDBULLETS PATH HERE"
        with open(path, 'r') as file:
            ds = json.load(file)
        questions = [ds['question'].values()]
        answer_choices = [{"A": ds['opa'][str(i)], "B": ds['opb'][str(i)], "C": ds['opc'][str(i)], "D": ds['opd'][str(i)]} for i in range(len(ds))]
        correct_answers = [ds['answer_idx'].values()]
        return questions, answer_choices, correct_answers

    elif task=="medbullets_op5":
        path = "ADD MEDBULLETS PATH HERE"
        with open(path, 'r') as file:
            ds = json.load(file)
        questions = [ds['question'].values()]
        answer_choices = [{"A": ds['opa'][str(i)], "B": ds['opb'][str(i)], "C": ds['opc'][str(i)], "D": ds['opd'][str(i)]} for i in range(len(ds))]
        correct_answers = [ds['answer_idx'].values()]
        return questions, answer_choices, correct_answers
    
    elif task=="pubmedqa":
        # This also contains context that is necessary for the question.
        path = "ADD PATH FOR PUBMEDQA HERE"

        with open(path, 'r') as file:
            ds = json.load(file)
        ds = list(ds.values())
        answer_choice_dict = {'A': "yes", 'B': "no", 'C': "maybe"}
        answer_choices = [answer_choice_dict]*len(ds)
        correct_answers = []
        questions = []
        for i in range(len(ds)):
            question_context = "Context: " + "\nContext: ".join(ds[i]['CONTEXTS'])
            questions.append(question_context + "\n" + ds[i]['QUESTION'])

            rev_answer_choice_dict = dict((v,k) for k,v in answer_choice_dict.items())
            answer = rev_answer_choice_dict.get(ds[i]['final_decision'])
            correct_answers.append(answer)
        return questions, answer_choices, correct_answers
    
    elif "mmlu" in task:
        subset = task.split("-", 1)[1]
        ds = load_dataset("cais/mmlu", subset, split=split)
        questions = [ds[i]['question'] for i in range(len(ds))]
        answer_choices = [{"A": ds[i]['choices'][0], "B": ds[i]['choices'][1], "C": ds[i]['choices'][2], "D": ds[i]['choices'][3]} for i in range(len(ds))]
        correct_answers = [chr(ds[i]['answer']+65) for i in range(len(ds))]
        return questions, answer_choices, correct_answers

    else:
        raise Exception("TASK NOT FOUND")

def filterContext(context):
    end_tag = "</end>"
    if end_tag in context:
        return context.split(end_tag)[0] + end_tag
    return context

def run_inference(content, engine, temp=0.0001, max_tokens_output=200, tokenizer=None, model=None, local=False, vllm = False):
    if local:
        messages = [{"role": "user", "content": f"{content}"}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda:0')
        outputs = model.generate(inputs, max_new_tokens=max_tokens_output, do_sample = True, temperature=temp)
        text = tokenizer.batch_decode(outputs)[0]
        return text.split("<|assistant|>")[-1]
    elif vllm:
        return None
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = [{"role": "user", "content": f"{content}"}]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens_output,
            frequency_penalty=0.0
        )
        response_text = response.choices[0].message.content
        return response_text
    
class MultiChoiceFilter:
    # Inspiring from lmeval
    def __init__(self, ignore_case=False, ignore_punctuation=False, regex_pattern=r"[\(\[]([A-Z])[\)\]]"):
        
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                                       if unicodedata.category(chr(i)).startswith("P"))

    def filter_text(self, text):
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(self.punct_tbl)
        return text

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict: 
                match = convert_dict[match]
        return match

    def extract_answer(self, response, choices=None):
        matchFirst = re.search(r'the answer is .(\w).', response)
        if matchFirst:
            return f"({matchFirst.group(1)})"
        match = self.find_match(self.regex, response) 
        if match:
            return f"({match})"
        return "[invalid]"

    def filter_responses(self, responses, choices):
        return [self.extract_answer(resp, choices) for resp in responses]