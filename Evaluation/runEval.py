from Evaluation.utils import *
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from Evaluation.prompts import *
import json
import pandas as pd
import pprint as p
import time
import random


def main():
    print("RUNNING NORMAL IMPLEMENTATION")
    ENGINE = "phi-3-mini-UltraMedical-NoLoRA"
    SPLIT = "test"
    NUMBER_OF_ENSEMBLE = 5
    if NUMBER_OF_ENSEMBLE > 1:
        ENGINE_TEMPERATURE = 0.7
    else:
        ENGINE_TEMPERATURE = 0.000000001 
    SCORE_THRESHOLD = 0
    MAX_TOKEN_OUTPUT = 1024
    MAX_NUMBER_OF_CONTEXT_PARA = 1
    SCORE_ENGINE = "phi-3" # [gpt-4o-mini, llama3.1, phi-3]
    SEARCH_ALGO = "None" ## For RAG... change to "None" if no context, ,["BM25", "RAG", "RAG_Title_LLM", "RAGBM25_take_top_5_each", "RAGBM25_lowest_index_sum", "RAGBM25_RRF"]
    NSHOT = 0
    STOP_GEN = 10000000 ## For testing purposes; stop generating after {STOP_GEN} amount of test-questions
    TASK_LIST = ['medmcqa'] # Options ["medqa", 'mmlu-anatomy', 'mmlu-professional_medicine', 'mmlu-college_biology', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-medical_genetics', pubmedqa, medmcqa"]
    OUTPUT_DIR = "OUTPUT DIRECTORY HERE"
    results_db = {
        "metadata": {
            "model" : ENGINE,
            "temperature" : ENGINE_TEMPERATURE,
            "num_shot" : NSHOT,
            "number_of_ensemble": NUMBER_OF_ENSEMBLE,
            "max_tokens" : MAX_TOKEN_OUTPUT,
        }
    }
    ## SET FILE DIRECTORY PATHS (Context dir, file output)
    if SEARCH_ALGO != "None":

        runName = f'MedMobile ({ENGINE}) COT simple prompt + {SEARCH_ALGO} Context scored by {SCORE_ENGINE}'
        if NUMBER_OF_ENSEMBLE >1:
            runName += f"+ Ensemble ({NUMBER_OF_ENSEMBLE})"
        contextdf_path = 'PATH TO RETRIEVE CONTEXT FROM'
        with open(contextdf_path, 'r') as file:
            contextdf = json.load(file)

        results_db['metadata']['context_search_algo'] = SEARCH_ALGO
        results_db['metadata']['score_engine'] = SCORE_ENGINE
        results_db['metadata']['score_threshold'] = SCORE_THRESHOLD
        results_db['metadata']['path_of_context'] = contextdf_path
        results_db['metadata']['number_of_context_paras'] = MAX_NUMBER_OF_CONTEXT_PARA
    else:   
        if NUMBER_OF_ENSEMBLE > 1:
            runName = f'MedMobile ({ENGINE}) + Ensemble ({NUMBER_OF_ENSEMBLE})'
        else: 
            runName = f'MedMobile ({ENGINE})'

    ## DISPLAY HYPERPARAMETERS
    for name, value in results_db['metadata'].items():
        print(f"{name} : {value}")

    ## LOAD IN MODEL IF LOCAL
    model_path = ENGINE
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda",torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        ENGINE,
        torch_dtype="auto",
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE)


    ## OUTPUT RUN INFO:
    print("Model Running: " + ENGINE)
    print("Run: " + runName)

    ## ASSIGN EVAL FILTER
    mcf = MultiChoiceFilter(ignore_case=True, ignore_punctuation=True)

    ## Process each task
    for task in TASK_LIST:
        question_list, answer_choices_list, correct_answer_list = task_load(task, SPLIT)
        print(f"{task} loaded succesfully. Now conducting evaluation on {len(question_list)} samples.")

        ## RECORD STARTING TIME & CREATE MODEL_DB
        start_time = time.time()
        model_db = []

        for i, (question, answer_choices, correct_answer) in tqdm(enumerate(zip(question_list, answer_choices_list, correct_answer_list))):
            D = {}
            context = ""
            context_counter = 0
            if SEARCH_ALGO != "None":
                for summary in contextdf['model_results'][i]['summary_list']:
                    if int(summary['regex_score']) >= SCORE_THRESHOLD:
                        context = context + summary['summary']
                        context_counter +=1
                        if context_counter == MAX_NUMBER_OF_CONTEXT_PARA:
                            break
                if NSHOT == 0:
                    prompt = prompt_eval_context_bare
                else: 
                    prompt = prompt_eval_with_context_and_examples
            else:
                if NSHOT == 0:
                    prompt = prompt_eval_bare_fully
                else: 
                    prompt = prompt_eval_bare_fully_with_examples

            ## Adjusting prompt depending on if there's few-shot ICL or not
            if NSHOT != 0:
                #examples = gpt_chain_of_thoughts['examples']
                #examples = sampleFromTrain(numShot)
                examples = extract_samples(task, NSHOT, prompt_example)
                model_prompt = prompt.format(
                    question=question,
                    choices=format_choices(answer_choices),
                    examples = ("\n").join(examples),
                    context = filterContext(context)
                )
            else:
                model_prompt = prompt.format(question=question, choices=format_choices(answer_choices), context = filterContext(context))

            ## Create question_dict that will eventually get added to master list of dict (model_db)
            D['query'] = question
            D['question_choices'] = answer_choices
            D['correct_answer'] = correct_answer
            D['attempts'] = []
            D['model_prompt'] = model_prompt

            for j in range(NUMBER_OF_ENSEMBLE):
                text = run_inference(model_prompt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, local=True)
                query_object = {'id': ('attempt_'+str(j)), 'COT': text}
                D['attempts'].append(query_object)
            model_db.append(D)

            if i == STOP_GEN-1:
                break

        end_time = time.time()
        total_num_ques = 0
        num_correct = 0
        num_invalid = 0
        for q in model_db: 
            choices = q['question_choices']
            letter_counts = {}
            for attempt in q['attempts']:
                attempt['model_choice'] = mcf.extract_answer(attempt['COT'], choices)
                if attempt['model_choice'] in letter_counts:
                    letter_counts[attempt['model_choice']] += 1 
                else:
                    letter_counts[attempt['model_choice']] = 1
            max_count = 0
            for letter, count in letter_counts.items():
                # if count > max_count and letter != "[invalid]":
                if count > max_count:
                    q['ensemble_answer'] = letter
                    max_count = count
            total_num_ques+=1
            if q['ensemble_answer'].strip("()") == q['correct_answer']:
                num_correct += 1
            elif q['ensemble_answer'] == "[invalid]":
                num_invalid += 1
        
        print("Number of correct answer: " + str(num_correct))
        print("Total number of questions: " + str(total_num_ques))
        print("Model Accuracy: " + str(num_correct/total_num_ques))
        
        results_db_task = results_db.copy()
        results_db_task['metadata']['informal_run_name'] = runName
        results_db_task['metadata']['task'] = task
        results_db_task['metadata']['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_db_task['metadata']['prompt'] = prompt
        results_db_task['metadata']['number_of_invalids'] = num_invalid
        results_db_task['metadata']['number_of_questions'] = total_num_ques
        results_db_task['metadata']['true_accuracy'] = num_correct/total_num_ques
        results_db_task['metadata']['eff_accuracy'] = num_correct/(total_num_ques-num_invalid)
        results_db_task['metadata']['run_time'] = end_time-start_time
        results_db_task['metadata']['run_time_per_iteration'] = (end_time-start_time)/total_num_ques
        results_db_task['model_results'] = model_db

        filename = f"{OUTPUT_DIR}{task}/query_database_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(results_db_task, file, indent=4)


if __name__ == "__main__":
    main()