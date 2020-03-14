# NSC4ExplainableAI

This repository contains an implementation of the work from "AI simbolica e sub-simbolica per XAI: stato dell'arte ed esperimenti con reti neurali e vincoli logici".

## Installation

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

python -m ntp.scripts.find_logic C:/Users/giuseppe.pisano/Documents/MyProjects/University/NSC4ExplainableAI/LogicInduction/NTP-improved/conf_synth/experiment_two.conf

## Credits

DL2 [[Website](https://www.sri.inf.ethz.ch/publications/fischer2019dl2)] [[pdf](https://files.sri.inf.ethz.ch/website/papers/icml19-dl2.pdf)]

NTP (Base) [[Website](https://github.com/uclmr/ntp)] [[pdf](https://arxiv.org/abs/1705.11040)]

NTP (Improved) [[Website](https://github.com/Michiel29/ntp-release)] [[pdf](https://arxiv.org/abs/1906.06805)]

tuProlog [[Website](http://apice.unibo.it/xwiki/bin/view/Tuprolog/WebHome)]
