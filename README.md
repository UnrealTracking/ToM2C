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

Note that the command above will load the default environment described in the paper. If you want to change the number of agents and targets, please refer to the `num-agents` and `num-targets` arguments.

After running the above command, you can run the following command respectively to do `Communication Reduction` mentioned in the paper:

```bash
python main.py --env MSMTC-v3 --model ToM2C --workers 6 --norm-reward --train-comm --load-model-dir [trained_model_file_path]
```

The above command is for cpu training. If you want to train the model on GPU, try to add `--gpu-id [cuda_device_id]` in the command. Note that this implementation does NOT support multi-gpu training.

## Rendering

After training, you can load the trained model and render its behavior by the following command.

In `CN`:

```bash
python render_test.py --env CN --model ToM2C --render --env-steps 10 --load-model-dir [trained_model_file_path]
```

In `MSMTC`:

```bash
python render_test.py --env MSMTC-v3 --model ToM2C --render --env-steps 20 --load-model-dir [trained_model_file_path]
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

