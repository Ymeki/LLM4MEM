import pandas as pd
import argparse
import time
import tqdm
import logging
import os
from datetime import datetime
from pathlib import Path
#from timer import Timer
from LLM4MEM.MPLAC.tools import *

if __name__ == '__main__':
    time_start = time.time()
    current_date = datetime.now().strftime("%Y%m%d")
    pfilepath = "/data/Shopee"
    base_dir_name = os.path.basename(pfilepath.rstrip(os.sep))  
    dated_subdir = f"{base_dir_name}-{current_date}"
    logsPath = os.path.join(pfilepath, dated_subdir, "log")
    datapath = Path(pfilepath)
    save_csv_dir = os.path.join(pfilepath, dated_subdir)
    init_log(logsPath)
    logging.info("start")
    logging.info(f"its time:{datetime.now()}")
    fileList = get_all_files(datapath, False)
    logging.info(f"find({len(fileList)})file")
    T, tables_df = read_all_tables(datapath)
    table_lens = [len(table) for table in tables_df] 
    table_ids = [table["tid"].tolist() for table in tables_df] 
    table_sentences = [textify_table(table) for table in tables_df]
    table_span = [vllm_batch(sentences,"llm") for sentences in zip(table_ids, table_sentences)]
    save_table_spans_to_csv(table_span, save_csv_dir)
    time_end = time.time()
    hour, minute, second = get_run_time(time_start, time_end)
    logging.info(f'run time:{hour}h{minute}m{second}s') 
