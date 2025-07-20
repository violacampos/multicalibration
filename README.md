# Masterarbeit

The scripts are tested on the hsrm megagpu server. It is possible that the results differ when executed on other operation systems/other configurations. This is the case because the data is stored in files and are read up on execution. Therefore the order of the read files can change.

## Structure
In the downloaded folder following folder can be found:
- masterarbeit
    - contains all scripts for calibration
- MultiPL-E
    - Benchmark used for code generation, with the extension of returning the token scores.

Furthermore the code generations are stored under 'MultiPL-E/run/'. Currently the generations were made with one model.


## Setup

All required python packages for the calibration approaches can be found in the requirement.txt file. You need to be in the masterarbeit folder for all following commands.

Therefore navigate to the downloaded folder and use:
```bash
cd masterarbeit
```


To install them create a venv and use the command:
```bash
pip install -r .\requirements.txt
```


## Analysis
To get an overview of the data streamlit can be used to explore individual examples.

Therfore execute the following command:
```bash
streamlit run analyse/sample_viewer.py 
```

## Commandline parameters

| Command | Description |
| --- | --- |
| --dir | Directory of the MultiPL-E code generations |
| --problem | Default: code-gen |
| --prob-method | Default: avg_logprob. Options: [avg_logprob, qualitativ, quantitativ] |
| --model | Options: [gpt_4o_mini]. Needs to be set for the qualitativ and quantitativ prob method |
| --bin-count| Default: 20. Number of bins to calibrate on |
| --grouping-style| Default: simple. Options: [simple, scc, categories, all] |
| --counter-groups | Enables the usage of counter groups |
| --save-table | Table will be saved in run folder |
| --save-charts | Table will be saved in run folder |
| --save-data | Saves the data of the comparison results |
| --split | By adding this the data is splitted in train, test, val |
| --k-fold | Option for run_ighb.py. Evaluates on 5-Fold split |
| --epsilon | Only effects run_iglb.py. Hyperparamter for early stopping. |
| --regressor | Default: LR. Options: [LR, SVR, XGBoost]. Only effects script run_regressor.py |
| --binning-type | Default: linear. Options: [linear]. Only effects script run_regressor.py |

## Comparison of methods
For a comparison of all methods the following command can be used:
```bash
python compare_methods.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --counter-groups --save-charts --save-table
```

This will output the results of every calibration approach. Further more a runs directory will be cerated where the charts are stored.

## Baseline
Command for calculating the baseline scores:
```bash
python compute_baseline.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --save-table --counter-groups
```
## Histogram binning
Command for using only the histogram binning approach:
```bash
python run_hb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table
```

## Linear regression
Command for using only the linear regression approach:
```bash
python run_lr.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table
```

## Iterative group histogram binning
Command for using only the iterative group histogram binning approach:
```bash
python run_ighb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table
```

To run the k-fold experiment on the IGHB approch the following command can be used:
```bash
python run_ighb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --k-fold --bin-count 100
```

## Iterative group linear binning
Command for using only the iterative group histogram binning approach:
```bash
python run_iglb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table
```

## Regressors
Command for using the differen regressor approach with linear regression:
```bash
python run_regressor.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table --model gpt_4o_mini --grouping-style all
```

Command for using the differen regressor approach with SVR:
```bash
python run_regressor.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1/ --prob-method avg_logprob --problem code-gen --split --save-charts --counter-groups --save-table --model gpt_4o_mini --grouping-style all --regressor SVR
```
