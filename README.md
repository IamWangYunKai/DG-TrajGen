# DG-TrajGen
The official repository for paper ''**Domain Generalization for Vision-based Driving Trajectory Generation'**' submitted to ICRA 2022.

[![arXiv](https://img.shields.io/badge/arXiv-2109.13858-B31B1B.svg)](https://arxiv.org/abs/2109.13858)
[![Project](https://img.shields.io/badge/Project-Site-orange.svg)](https://sites.google.com/view/dg-traj-gen/)
[![YouTube](https://img.shields.io/badge/YouTube-Video-green.svg)](https://www.youtube.com/watch?v=hvuUtPz8U24&t=9s)
[![Bilibili](https://img.shields.io/badge/Bilibili-Video-blue.svg)](https://www.bilibili.com/video/BV1AQ4y167hc?spm_id_from=333.999.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Our Method
![structure](./imgs/structure.png)

* **Trajectory representation**:
  * Model: ./learning/model.py/Generator
* **Latent Action Space Learning**:
  * Generator model: ./learning/model.py/Generator
  * Discriminator model: ./learning/model.py/Discriminator
  * Training: ./scripts/Ours/stage1_train_GAN.py
* **Encoder Pre-training**:
  * Training: ./scripts/Ours/stage2_pretrain_encoder.py
* **End-to-End Training the Encoder**:
  * Training: ./scripts/Ours/stage3_train_e2e.py

## Comparative Study
* RIP:
  * Training: ./scripts/RIP/train.py
  * Referenced official code: [github](https://github.com/OATML/oatomobile/)
  * Paper: [arxiv](https://arxiv.org/abs/2006.14911)
* MixStyle:
  * Training: ./scripts/MixStyle/train.py
  * Referenced official code: [github](https://github.com/KaiyangZhou/mixstyle-release)
  * Paper: [arxiv](https://arxiv.org/abs/2104.02008)
* DIVA:
  * Training: ./scripts/DIVA/train.py
  * Referenced official code: [github](https://github.com/AMLab-Amsterdam/DIVA)
  * Paper: [arxiv](https://arxiv.org/abs/1905.10427)
* DAL:
  * Training: ./scripts/DAL/train.py
* E2E NT:
  * Training: ./scripts/E2ENT/train.py
  * Referenced official code: [github](https://github.com/ZJU-Robotics-Lab/CICT)
  * Paper: [arxiv](https://arxiv.org/abs/2010.10393)

![comp](./imgs/comp.png)
## Closed-loop Experiments:
We train the model on the Oxford RobotCar dataset and **directly generalize** it to the CARLA simulation.
* Run: ./scripts/CARLA/run_ours.py

![carla](./imgs/carla.png)

![ClearNoon](./imgs/ClearNoon.gif)
![WetCloudySunset](./imgs/WetCloudySunset.gif)
![HardRainSunset](./imgs/HardRainSunset.gif)
![HeavyFogMorning](./imgs/HeavyFogMorning.gif)


## Citation
If you use our source code, please consider citing the following:
```bibtex
@article{wang2021domain,
  title={Domain Generalization for Vision-based Driving Trajectory Generation},
  author={Wang, Yunkai and Zhang, Dongkun and Cui, Yuxiang and Chen, Zexi and Jing, Wei and Chen, Junbo and Xiong, Rong and Wang, Yue},
  journal={arXiv preprint arXiv:2109.13858},
  year={2021}
}
```