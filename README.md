# ToM2C

This repository is the offcial implementation of ToM2C, "[ToM2C: Target-oriented Multi-agent Communication and Cooperation with Theory of Mind (ICLR 2022)](https://arxiv.org/abs/2111.09189)" . 

## Installation

To install requirements:

```bash
pip install -r requirements.txt
```

All the environments have been included in the code, so there is no need to install Multi-sensor Multi-target Coverage(MSMTC) or MPE(Cooperative Navigation) additionally.

## Training

To train ToM2C in `MSMTC`, run this command:

```bash
python main.py --env MSMTC-v3 --model ToM2C --workers 6 --norm-reward
```

To train ToM2C in `CN`, run this command:

```bash
python main.py --env CN --model ToM2C --workers 6 --env-steps 10 --A2C-steps 10 --norm-reward
```

## Citation

If you found ToM2C useful, please consider citing:
```
@inproceedings{
wang2021tomc,
title={ToM2C: Target-oriented Multi-agent Communication and Cooperation with Theory of Mind},
author={Yuanfei Wang and Fangwei Zhong and Jing Xu and Yizhou Wang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=M3tw78MH1Bk}
}
```
## Contact

If you have any suggestion or questions, please get in touch at [yuanfei_wang@pku.edu.cn](yuanfei_wang@pku.edu.cn) or [zfw@pku.edu.cn](zfw@pku.edu.cn).

