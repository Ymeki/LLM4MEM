python -m vllm.entrypoints.openai.api_server  \
    --served-model-name llm \
    --model Qwen2.5_7B \
    --tensor-parallel-size 1 \
    --port 8016