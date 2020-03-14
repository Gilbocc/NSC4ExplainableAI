# NSC4ExplainableAI

This repository contains an implementation of the work from "AI simbolica e sub-simbolica per XAI: stato dell'arte ed esperimenti con reti neurali e vincoli logici".

## Installation

**Tensorflow GPU** library is required to run the project correctly. You can find all the details on the needed requirements [here](https://www.tensorflow.org/install/gpu).

0 - Install Anaconda

1 - Create the virtual environment
    
    conda create --prefix=./envs python=3.6

2 - Activete the newly created environment
    
    conda activate ./envs

3 - Install the dependencies
    
    python ./install.py

4 - To disable the virtual environment use
    
    conda deactivate

5 - To delete the envireonment use

    conda env remove -p ./envs

## Usage

### Experiment one (**unbalance**)

You need to replace *ACTUAL_PATH* with the absolute path to the folder where the project has been cloned 

Model/dataset to Prolog theory translation

    python ./main/dataset/experiment_one/theory_generator.py --model_path ACTUAL_PATH/main/dataset/experiment_one/output_model.ph --dataset_path ACTUAL_PATH/main/dataset/experiment_one/dataset_final.csv --theory_path ACTUAL_PATH/main/dataset/experiment_one/theory.pl --is_model False/True

Rules induction

    python ./main/induction/find_logic.py --config_path ACTUAL_PATH/main/induction/config/experiments_config.conf --theory_path ACTUAL_PATH/main/dataset/experiment_one/theory.pl --rules_template_path ACTUAL_PATH/main/dataset/experiment_one/rules_templates.pl --rules_path ACTUAL_PATH/main/dataset/experiment_one/rules.pl

Network training (*No constraining*)

    python ./main/network/experiment_one.py --path ACTUAL_PATH\main\dataset\experiment_one\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_one\output_model.ph --save_output True --constraint_weight 0.0 --global_constraining False --num_epochs 50 --random_seed_base 41 --num_runs 1

Network training (*Local constraining*)

    python ./main/network/experiment_one.py --path ACTUAL_PATH\main\dataset\experiment_one\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_one\output_model.ph --save_output True --constraint_weight 0.7 --global_constraining False --num_epochs 50 --random_seed_base 41 --num_runs 1

Network training (*Global constraining*)

    python ./main/network/experiment_one.py --path ACTUAL_PATH\main\dataset\experiment_one\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_one\output_model.ph --save_output True --constraint_weight 0.0 --global_constraining True --num_epochs 10 --random_seed_base 41 --num_runs 1

### Experiment two (**dimensionality**)

Model/dataset to Prolog theory translation

    python ./main/dataset/experiment_two/theory_generator.py --model_path ACTUAL_PATH/main/dataset/experiment_two/output_model.ph --dataset_path ACTUAL_PATH/main/dataset/experiment_two/dataset_final.csv --theory_path ACTUAL_PATH/main/dataset/experiment_two/theory.pl --is_model False/True

Rules induction

    python ./main/induction/find_logic.py --config_path ACTUAL_PATH/main/induction/config/experiments_config.conf --theory_path ACTUAL_PATH/main/dataset/experiment_two/theory.pl --rules_template_path ACTUAL_PATH/main/dataset/experiment_two/rules_templates.nlt --rules_path ACTUAL_PATH/main/dataset/experiment_two/rules.pl

Network training (*No constraining*)

    python ./main/network/experiment_two.py --path ACTUAL_PATH\main\dataset\experiment_two\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_two\output_model.ph --save_output True --constraint_weight 0.0 --global_constraining False --num_epochs 100 --random_seed_base 41 --num_runs 1

Network training (*Local constraining*)

    python ./main/network/experiment_two.py --path ACTUAL_PATH\main\dataset\experiment_two\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_two\output_model.ph --save_output True --constraint_weight 0.2 --global_constraining False --num_epochs 100 --random_seed_base 41 --num_runs 1

Network training (*Global constraining*)

    python ./main/network/experiment_two.py --path ACTUAL_PATH\main\dataset\experiment_two\dataset_final.csv --model_path ACTUAL_PATH\main\dataset\experiment_two\output_model.ph --save_output True --constraint_weight 0.0 --global_constraining True --num_epochs 40 --random_seed_base 41 --num_runs 1

## Credits

DL2 [[Website](https://www.sri.inf.ethz.ch/publications/fischer2019dl2)] [[pdf](https://files.sri.inf.ethz.ch/website/papers/icml19-dl2.pdf)]

NTP (Base) [[Website](https://github.com/uclmr/ntp)] [[pdf](https://arxiv.org/abs/1705.11040)]

NTP (Improved) [[Website](https://github.com/Michiel29/ntp-release)] [[pdf](https://arxiv.org/abs/1906.06805)]

tuProlog [[Website](http://apice.unibo.it/xwiki/bin/view/Tuprolog/WebHome)]
