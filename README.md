# FurnitureBench: Reproducible Real-World Furniture Assembly Benchmark

[**Paper**](http://arxiv.org/abs/2305.12821)
| [**Website**](https://clvrai.com/furniture-bench/)
| [**Documentation**](https://clvrai.github.io/furniture-bench/docs/index.html)

![FurnitureBench](furniture_bench_banner.jpg)

FurnitureBench is the real-world furniture assembly benchmark, which aims at providing a reproducible and easy-to-use platform for long-horizon complex robotic manipulation.

It features
- Long-horizon complex manipulation tasks
- Standardized environment setup
- Python-based robot control stack
- FurnitureSim: a simulated environment
- Large-scale teleoperation dataset (200+ hours)

Please check out our [website](https://clvrai.com/furniture-bench/) for more details.


## FurnitureBench

We elaborate on the real-world environment setup guide and tutorials in our [online document](https://clvrai.github.io/furniture-bench/docs/index.html).


## FurnitureSim

FurnitureSim is a simulator based on Isaac Gym. FurnitureSim works on Ubuntu and Python 3.8+. Please refer to [Installing FurnitureSim](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html) and [How to Use FurnitureSim](https://clvrai.github.io/furniture-bench/docs/tutorials/furniture_sim.html) for more details about FurnitureSim.


## Citation

If you find FurnitureBench useful for your research, please cite this work:
```
@inproceedings{heo2023furniturebench,
    title={FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation},
    author={Minho Heo and Youngwoon Lee and Doohyun Lee and Joseph J. Lim},
    booktitle={Robotics: Science and Systems},
    year={2023}
}
```


## References

- Polymetis: https://github.com/facebookresearch/polymetis
- BC: Youngwoon's [robot-learning repo](https://github.com/youngwoon/robot-learning).
- IQL: https://github.com/ikostrikov/implicit_q_learning
- R3M: https://github.com/facebookresearch/r3m
- VIP: https://github.com/facebookresearch/vip
- Factory: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md
- OSC controller references: https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic
