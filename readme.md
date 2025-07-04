
# LLM4MEM
🎉 This paper has been accepted as the NLPCC 2025 poster paper!
LLM4MEM: Unlocking the Power of Large Lanaguage Models for Multi-Table Entity Matching, an unsupervised framework that utilizes a Large Language Model (LLM) for multi table matching, with three key innovations: 
1) multi-style prompt-enhances attribute coordination through semantic regularization supported by LLM to dynamically align heterogeneous patterns.
2) constraint-propagation embedding matching optimization reduces comparison complexity to O(n) through bidirectional top-1 filtering and graph propagation, and 
3) density-aware pruning eliminates noise and false positive information through spatial constraint optimization.

## How 2 Start
--
creat a new env and pip Requirements:
```
pyhton-sklearn,pandas,numpy,sentence-transformers,networkx,joblib,hnswlib,tyro,pyfunctional,loguru,vllm,openAI,modelscope
```

-- download LLM model
```py
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen-7B', local_dir='infor of LLM')
```
-- dataset
```
download from
https://drive.google.com/drive/folders/1ymzl7MDuaJLv5V_TNqcc8ooR4V2uW4g1?usp=drive_link
```
--MPLAC
```sh
sh run.sh
python MPLAC/submain.py
```
--TCEM and PDM
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --name Music-20
```

