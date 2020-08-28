# Kernel_learning_distribution
## Supplement
In this paper, the first synthetic test data is composed of the following two spectrally-overlapped components:  
\begin{align}
\begin{cases}
x_{1}(t)=a(t)\cos \left\{ 2\pi \left( 0.0006(t^{2}-t_{0}^{2})+0.15(t-t_{0}) \right) \right\},  \\
x_{2}(t)=a(t)\cos \left\{ 2\pi \left( -0.0006(t^{2}-t_{0}^{2})+0.35(t-t_{0}) \right) \right\},
\end{cases}\notag
\end{align}

where $a(t)=\exp\big(-(0.005t-0.667)^{2}\pi\big)$ is an Gaussian AM function, $t_0$ equals to 128.
## Preparation
- python 3.6
- pytorch 0.4.1
- cuda 9.0
- cudNN 7.6.3.30
## Contributing Guideline
We would like to thank the authors in these works for sharing the source codes of these methods, which are publicly released at https://github.com/Prof-Boualem-Boashash/TFSAP-7.1-software-package and https://github.com/mokhtarmohammadi/Locally-Optimized-ADTFD.
## Contact
This repo is currently maintained by Lei Jiang (teki97@whu.edu.cn).
