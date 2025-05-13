# Masterarbeit

## Commands

Add before script execution to select Graphics Card('s) <br/>
`CUDA_VISIBLE_DEVICES=7`<br/>
`CUDA_VISIBLE_DEVICES=4,5,6,7`

Caculates the ECE/Brier score for the Benchmarks in the runs directory (you can also select single runs)<br/>
`python calculate_ece.py ../MultiPL-E/runs/`

Creates charts for the given run (reliability chart, (total, pass, fail) propability distribution)<br/>
`python generate_charts_multipl_e.py ../MultiPL-E/runs/mbpp-py-keep-_data_tyler_llms_llama3.1_huggingface_Meta_Llama_3.1_8B_Instruct_-1.0-reworded/`

Runs the generation Process of the MultiPL-E Benchmark (--use-local and dataset) or (--lang rs to chose language).<br/>
Set --batch-size to 10 for 70B Models.<br/>
`python automodel_vllm.py --name Qwen/Qwen2.5-Coder-7B-Instruct \
                          --root-dataset humaneval \
                          --use-local \
                          --dataset prompts/mbpp-py-keep.jsonl \
                          --temperature 1.0 \
                          --batch-size 20 \
                          --completion-limit 1 \
                          --output-dir-prefix runs`

use generate_4_all script to generate code for all different languages. Parameters are specified in the script.
CUDA_VISIBLE_DEVICES=6,7 python generate_4_all.py --use-local

After generation of samples. use the Multipl_E script to check for correctness. Copy samples to local machin in the Multipl-E repo
under the folder runs.
Then excute the following command:
docker run --rm --network none -v G:/Masterarbeit/MultiPL-E/runs:/runs:rw multipl-e-eval --dir /runs/humaneval-all-keep-Qwen2.5_Coder_14B-Instruct-1.0-comp-1 --output-dir /runs/humaneval-all-keep-Qwen2.5_Coder_14B-Instruct-1.0-comp-1 --recursive                    

replace dir with the name of the copied folder

After the evaluation copy the folder onto the hostmachine if necessary.

Run the script create programs to obtain information about the generated code. But before that adjust the run parameter in the script.

After programs are created install scc by boyter and execute the following command.
scc -f json -o results.json --by-file programs

this will generat a json file with informations about every program. To use them copy them into the scc folder in the project masterarbeit and rename it to the name of the run.

Now you can run the compare_methods script to get a result for all methods:
python compare_methods.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1 --split --use-scc --prob-method avg_logprob --save-table

python compare_methods.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_14B-Instruct-1.0-comp-1 --use-scc --prob-method avg_logprob --save-table --save-charts --split --save-data

