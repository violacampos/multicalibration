# Masterarbeit

## Commands

Add before script execution to select Graphics Card('s) <br/>
`CUDA_VISIBLE_DEVICES=7`
`CUDA_VISIBLE_DEVICES=4,5,6,7`

Caculates the ECE/Brier score for the Benchmarks in the runs directory (you can also select single runs)<br/>
`python calculate_ece.py ../MultiPL-E/runs/`

Creates charts for the given run (reliability chart, (total, pass, fail) propability distribution)<br/>
`python generate_charts_multipl_e.py ../MultiPL-E/runs/mbpp-py-keep-_data_tyler_llms_llama3.1_huggingface_Meta_Llama_3.1_8B_Instruct_-1.0-reworded/`

Runs the generation Process of the MultiPL-E Benchmark (--use-local and dataset) or (--lang rs to chose language).<br/>
Set --batch-size to 10 for 70B Models.
`python automodel_vllm.py --name Qwen/Qwen2.5-Coder-7B-Instruct \
                          --root-dataset humaneval \
                          --use-local \
                          --dataset prompts/mbpp-py-keep.jsonl \
                          --temperature 1.0 \
                          --batch-size 20 \
                          --completion-limit 1 \
                          --output-dir-prefix runs`

