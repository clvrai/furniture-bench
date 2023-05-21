# FurnitureBench: Reproducible Real-World Furniture Assembly Benchmark

[**Paper**](#)
| [**Website**](https://clvrai.com/furniture-bench/)
| [**Documentation**](https://clvrai.github.io/furniture-bench/docs/index.html)

![FurnitureBench](readme_img/banner.jpg)

FurnitureBench is the real-world furniture assembly benchmark, which aims at providing a reproducible and easy-to-use platform for long-horizon complex robotic manipulation.

It features
- Long-horizon complex manipulation tasks
- Standardized environment setup
- Python-based robot control stack
- FurnitureSim: a simulated environment
- Large-scale teleoperation dataset (200+ hours)

Please check out our [website](https://clvrai.com/furniture-bench/) for more details.


## Documentation generation

The source files of our online documentation are under `docs_source/`.
If you want to build the online documentation, try the following steps:
```
pip install -r requirements.txt
./build_and_move.sh
```
`build_and_move.sh` builds the documentation and moves the outputs to `docs/` directory.


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
