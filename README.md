# ToM2C

This repository implements ToM2C, which is the codebase of the paper "[ToM2C: Target-oriented Multi-agent Communication and Cooperation with Theory of Mind](https://arxiv.org/abs/2111.09189)"(ICLR 2022). 

## Installation

To install requirements:

```bash
pip install -r requirements.txt
```

All the environments have been included in the code, so there's no need to install Multi-sensor Multi-target Coverage(MSMTC) or MPE(Cooperative Navigation) additionnaly.

## Run

To train ToM2C in MSMTC, run this command:

```bash
python main.py --env Pose-v3 --model ToM-v5 --workers 6 --norm-reward
```

The main branch is the code of ToM2C in MSMTC environment. As for Cooperative Navigation, please change to the CN branch.

## Contact

If you have any suggestion/questions, email yuanfei_wang@pku.edu.cn.

