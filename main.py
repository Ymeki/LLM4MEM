from itertools import chain
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from args import build_main_args
from data import Table, read_all_tables, textify_table, read_ground_truth
from log import init_logger, log_args, log_time, log
from timer import Timer
from metrics import evaluate_log
from selector import auto_selection
from merger import merge, merge_parallel
from pruner import pruning, pruning_parallel
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def diff(d1, d2):
    set_d1 = set(d1)
    set_d2 = set(d2)
    intersection = set_d1 & set_d2
    new_d1 = list(set_d1 - intersection)
    new_d2 = list(set_d2 - intersection)
    
    return new_d1, new_d2

def csls(similarity_matrix, k=3):
    sim_en = np.sort(similarity_matrix, axis=1)[:, -k:]
    sim_de = np.sort(similarity_matrix, axis=0)[-k:, :]
    
    sim_en = np.mean(sim_en, axis=1)
    sim_de = np.mean(sim_de, axis=0)
    adjusted_sim_matrix = 2 * similarity_matrix - sim_en[:, np.newaxis] - sim_de[np.newaxis, :]
    return adjusted_sim_matrix

def csls_sim(sim_mat, k): 
    """
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters 
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
    csls_sim_mat = 2 * sim_mat.t() - nearest_values1
    csls_sim_mat = csls_sim_mat.t() - nearest_values2
    return csls_sim_mat


if __name__ == '__main__':
    args = build_main_args()
    file_name = f"main"
    log_file_name = init_logger(file_name)
    log_args(args)
    log(log_file_name)
    data_path = Path(args.data_path)
    full_data_path = data_path / args.name
    timer = Timer()
    T, tables_df = read_all_tables(full_data_path)
    tm = timer.stop()
    table_lens = [len(table) for table in tables_df]
    n = sum(table_lens)
    table_ids = [table["tid"].tolist() for table in tables_df]
    tables = [Table(str(idx), table_id, list(range(len(table_id))))
              for idx, table_id in enumerate(table_ids)]
    timer.start()
    table_sentences = [textify_table(table) for table in tables_df]
    model = SentenceTransformer(args.lm_model_or_path)
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    table_embeddings = [model.encode(sentences, show_progress_bar=True, batch_size=args.batch_size, normalize_embeddings=True) for sentences in table_sentences]
    all_embeddings = list(chain(*table_embeddings))
    all_embeddings = np.array(all_embeddings)
    tm = timer.stop()
    ground_truth = read_ground_truth(full_data_path)
    timer.start()
    table = merge(tables, all_embeddings, args)
    tm = timer.stop()
    prediction = table.get_tuples()
    evaluate_log(ground_truth, prediction)
    timer.start()
    new_prediction = pruning(prediction, all_embeddings, args)
    tm = timer.stop()
    evaluate_log(ground_truth, new_prediction)
