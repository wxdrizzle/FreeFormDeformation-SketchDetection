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
python3 -m pip install -r requirements.txt 
# For faster installation, Chinese users may use:
#python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Sometimes, Mac OS users may encounter an error `Failed building wheel for pycairo`. This is a problem related to the installation of pycairo rather than our project. A typical solution is to install pycairo via conda:

```
# Reference: https://anaconda.org/conda-forge/pycairo
conda install -c conda-forge pycairo
```

## Using Software

Try running the following after setting working directory:
```sh
python GUI.py
```
## License

This repository is under the MIT license.
