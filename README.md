<img src="README.assets/GUI_image.png" alt="GUI_image" style="zoom:25%;" />

# Free Form Deformation (FFD) & Sketch Detection

Authors: Xin Wang, Xiaoyu Zhou, Danran Chen

This repository contains the final project of the authors in the course Data Visualization (2020 Fall) at Fudan University.

[Report PDF](https://github.com/lsDrizzle/FreeFormDeformation-SketchDetection/blob/main/%E5%9B%BE%E5%83%8F%E5%AE%9E%E6%97%B6%E9%9D%9E%E5%88%9A%E6%80%A7%E5%BD%A2%E5%8F%98%E5%92%8C%E8%8D%89%E5%9B%BE%E7%94%9F%E6%88%90.pdf)   [Demo Video](https://github.com/lsDrizzle/FreeFormDeformation-SketchDetection/blob/main/GUI%E6%BC%94%E7%A4%BA.mov)

## Highlights

- We implement an accelerated version of the FFD algorithm so as to warp images **in real time**, in which a reverse mapping takes only **0.015 seconds** and a deformation takes only **0.1 seconds** on i5-8279U@2.4GHz without GPU acceleration.
- We offer **four types of animations** for space transformation visualization, using the [Manim](https://github.com/3b1b/manim) library.
- We implement **six types of edge detection** algorithms, i.e., Sobel, Canny, Laplacian, LoG, DoG, and XDoG, for sketch generation.
- All of our algorithms are implemented directly using Numpy, Pytorch (for computing derivatives automatically) and other libraries at the same level, rather than some professional image processing libraries like OpenCV, which provide too many black-box functions.

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
