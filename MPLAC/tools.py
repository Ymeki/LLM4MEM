import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger
from typing import List, Tuple
from dataclasses import dataclass
# from functional import pseq
import json
import re
import requests
import concurrent.futures
import time
from functools import partial
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from LLM4MEM.MPLAC.prompts import prompt

# from callLLM import callLLM
from vllms import callLLM_batch

def init_log(log_dir):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"log_{current_time}.txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        filename=log_filepath,  
        level=logging.INFO,      
        format="%(asctime)s - %(levelname)s - %(message)s",  
        datefmt="%Y-%m-%d %H:%M:%S" 
    )
    print("logs: %s", log_filepath)

def get_run_time(time_start, time_end):
    hour = (time_end - time_start)//3600
    minute = ((time_end - time_start)-3600*hour)//60
    second = (time_end - time_start)-3600*hour-60*minute
    return hour, minute, second

def log(msg):
    logger.info(msg)

def log_time(desc: str, elapsed_time: float):
    logger.info(f"[{desc}]: {elapsed_time:0.4f} seconds")

def get_all_files(dirpath, ifdir = False, file_extension='.csv'):
    files_found = []
    if ifdir:
        for root, _, files in os.walk(dirpath):
            for file in files:
                if file.endswith(file_extension):
                    file_path = os.path.join(root, file)
                    files_found.append(file_path)
    else:
        for file in os.listdir(dirpath):
            file_path = os.path.join(dirpath, file)
            if os.path.isfile(file_path) and file.endswith(file_extension):
                files_found.append(file_path)
    
    return files_found

def read_table(data_path: Path, selected_attrs=None):
    if selected_attrs is None:
        table = pd.read_csv(
            data_path, dtype={"postcode": str})
    else:
        table = pd.read_csv(
            data_path, dtype={"postcode": str}, usecols=selected_attrs)
    return table


def read_all_tables(data_path: Path, num=-1, selected_attrs=None) -> Tuple[int, List[pd.DataFrame]]:
    i = 0
    tables = []
    while (data_path / f"table_{i}.csv").is_file():
        table = read_table(data_path / f"table_{i}.csv", selected_attrs)
        tables.append(table)
        i += 1
        if i == num:
            break
    return i, tables
@dataclass()
class Table:
    idx: str
    tids: List[int]
    tuple_ids: List[int]

    def get_tuples(self, min_cnt=1):
        res = pseq(zip(self.tids, self.tuple_ids))\
            .group_by(lambda x: x[1])\
            .map(lambda x: x[1])\
            .map(lambda x: [xi[0] for xi in x])\
            .filter(lambda x: len(x) > min_cnt)\
            .map(lambda x: sorted(x))\
            .map(lambda x: tuple(x))\
            .to_list()
        return res
    
def textify_table(table: pd.DataFrame):#
    sentences = table.iloc[:,1:] \
        .astype(str) \
        .apply(lambda x: x + ",") \
        .values.sum(axis=1)\
        .tolist()
    return sentences

def savefile(list, outputpath):
    dirpath = os.path.dirname(outputpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    try:
        with open(outputpath, 'w', encoding='utf-8') as f:
            for line in list:
                json.dump(line, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"save {outputpath} wrong:{e}")


def span_list_batch(listdata, model='Qwen', batch_size=50):
    span_list = []
    
    # Split the list into batches
    sentences_batches = [listdata[1][i:i + batch_size] for i in range(0, len(listdata[1]), batch_size)]
    idx_batches = [listdata[0][i:i + batch_size] for i in range(0, len(listdata[0]), batch_size)]
    
    sentences_batches = [i for i in listdata[1]]
    idx_batches = [i for i in listdata[0]]
    isfirst = True
    for idx_batch, sentence_batch in tqdm(zip(idx_batches, sentences_batches), desc="Adding spans...", total=len(idx_batches)):
        cetencespan = callLLM_batch(sentence_batch, model, isfirst)
        isfirst = False
        try:
            # Call the model in batch
            for idx, sentence in zip(idx_batch, cetencespan):
                data_json = {"tid": idx}
                fixed_str = json.loads(sentence)
                data_json.update(fixed_str)
                span_list.append(data_json)
        except json.JSONDecodeError as e:
            for idx in idx_batch:
                json_data = {"tid": idx, "wrong_info": str(e)}
                span_list.append(json_data)
                logging.warning(f"{e}<---->{idx}")
    
    return span_list


def vllm_batch(listdata, model="llm"):
    idx_batches, sentences_batches = listdata[0], listdata[1]
    results = []
    if len(idx_batches) != len(sentences_batches):
        raise ValueError("lenth is wrong")
    total_items = len(idx_batches) 
    completed = 0  
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = ((idx, sentence, model) 
                for idx, sentence in zip(idx_batches, sentences_batches))
        futures = [executor.submit(process_single_item, *task) for task in tasks]
        with tqdm(total=total_items, desc="Processing", unit="item") as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"error": f"wrong: {str(e)}"})
                pbar.update(1)
    results_sorted = sorted(results, key=lambda x: x.get("tid", float("inf")))
    return results_sorted

def process_single_item(idx, sentence, model):
    llm_response = get_LLM_response_vllm(sentence, model)
    if llm_response == "Err...":
        return {"tid": idx, "error": "LLM false"}
    try:
        response_data = json.loads(llm_response)
        return {"tid": idx, **response_data}
    except json.JSONDecodeError as e:
        logging.warning(f"JSON false tid:{idx} - {str(e)}")
        return {"tid": idx, "error": "JSON false", "raw": llm_response}
    except Exception as e:
        logging.error(f"false tid:{idx} - {str(e)}")
        return {"tid": idx, "error": str(e)}

def get_LLM_response_vllm(item, model="llm"):
    #conversation = [{"role":"user", "content": prompt}]
    conversation = [
            {
                "role": "system",
                "content": prompt["shopee-desc"]
            },
            {
                "role": "user",
                "content": item,
            }
        ]
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8016/v1"

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    delay = 3
    i = 0
    maxtry = 20
    while i < maxtry:
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=conversation,
                n=1, 
                max_tokens=256,
                temperature=0.0, 
                top_p=0.95
            )
            return chat_response.choices[0].message.model_dump()['content']
        except Exception as e:
            i += 1
            print(f"Try {i + 1} False: {e} {delay} sec retry...")
            time.sleep(delay)
            continue
    return "Err..."


def generate_text(data, model='Qwen'):
    server_url = "http://localhost:8000/generate"
    request_data = {
        "data": data,
        "model": model
    }
    try:
        response = requests.post(server_url, json=request_data)
        response.raise_for_status()
        result = response.json()
        return result.get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"false:{e}")
        return []


def save_table_spans_to_csv(table_span, output_dir="output_csvs"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"save CSV: {output_dir}")
    for idx, spans in enumerate(table_span):
        if not spans:
            logging.info(f"false: table {idx}")
            continue
        df = pd.DataFrame(spans)
        file_name = f"table_{idx}.csv"
        file_path = os.path.join(output_dir, file_name)
        try:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logging.info(f"save {idx} to: {file_path}")
        except Exception as e:
            logging.info(f"false: cant save {idx} 2 {file_path}wrong: {e}")
