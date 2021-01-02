<img src="README.assets/GUI_image.png" alt="GUI_image" style="zoom:25%;" />

# Free Form Deformation (FFD) & Sketch Detection

Authors: Xin Wang, Xiaoyu Zhou, Danran Chen

This repository contains the final project of the authors in the course Data Visualization (2020 Fall) at Fudan University.

## Installation

The software is verified to be operational on: 

- Mac OS Big Sur 11.1 
- Python 3.7

There is no guarantee that it will run successfully or that the interface will stay the same in other environments (such as Windows 10), although it should in theory.

We strongly recommend using Anaconda. You may follow the following steps:

#### 1. Create and navigate to a new environment

```bash
 conda create -n FFD python=3.7
 conda activate FFD
```

#### 2. Install prerequisites

Prerequisite information can be found in `requirements.txt `. You may use pip to install them:

```bash
python -m pip install -r requirements.txt
# For faster installation, Chinese users may add the "-i" option when runing pip:
# python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Troubleshooting

#### 1. `Failed building wheel for pycairo`

This error may occur while you are installing Pycairo by runing `python -m pip install -r requirements.txt`. This is a problem related to the installation of Pycairo rather than our software. A typical solution is to install Pycairo via conda:

```bash
# Reference: https://anaconda.org/conda-forge/pycairo
conda install -c conda-forge pycairo
```

#### 2. `Class QMacAutoReleasePoolTracker is implemented in both ...`

This error usually occurs when running our software on Mac OS. It is due to the library conflicts between OpenCV and PyQT5. To solve this problem, first uninstall current OpenCV:

```bash
pip uninstall opencv-python
# Or, you may need to try this: pip uninstall opencv-contrib-python
```

Then, install the headless version:

```
pip install opencv-contrib-python-headless
```

That is one of the reasons why we recommend creating a new environment using Anaconda.

## Using Software

Try running the following after setting working directory:
```sh
python GUI.py
```
## Acknowledgments

We would like to thank Yuncheng Zhou (master student in the School of Data Science at Fudan) for his instruction in FFD algorithms.

We would like to thank the Python library [Manim](https://github.com/3b1b/manim). We modified the source codes it provides for customizing mathematical animations.

## License

This repository is under the MIT license.
