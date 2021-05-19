# Kernel Learning for High-Resolution Time-frequency Distribution
![](https://github.com/teki97/KL-TFD/blob/main/supplement1.png)
We provide a pytorch implementation of the paper: Kernel Learning for High-Resolution Time-Frequency Distribution [1], where a kernel learning based time-frequency distribution (TFD) model is proposed to gain high resolution and CT-free TFDs. As shown in the above figure, the proposed model includes **N** Skipping Weighted Conv Modules. Specifically, several stacked multi-channel learning convolutional kernels to simulate adaptive directional filters while skipping operator is utilized to maintain correct information transmission. In addition, bottleneck attention module (BAM) [2] with groupnormalization is regarded as the weighted block to improve resolution.  
All pre-trained network related to this paper are provided. The training code will be provided soon.

## Preparation
- python 3.6
- pytorch 0.4.1
- cuda 9.0
- cudNN 7.6.3.30

## Training Dataset
Our training dataset includes two or three randomly spectrally-overlapped (only one intersection) linear frequencymodulated (LFM) and sinusoidal frequency-modulated (SFM) components with amplitude modulation (AM) at a fixed SNR = **10 dB**. In particular, two-component synthetic data is composed of a LFM and a SFM, and three-component synthetic data is composed of two LFMs and a SFM with only one intersection between a LFM and a SFM. All synthetic data has 256 samples. Each epoch randomly generates batchsize=16 synthetic signals for training by changing the slope of the LFM signal and the smallest and normalized frequencies of the SFM signal. The proportion for two and three-component signals is 50% and 50%. Test synthetic signals in this paper is not in our training dataset.  
For the specific application, you'd better make the synthetic data in the training dataset fit to the real-life signals to get a satisfactory performance.

## Supplementary

### Discussion on Real-life Data
In this paper, we discuss the robustness of our network on synthetic data. We also have some discussion on real-life data corresponding to various **N** (4, 6, 8, 10, 12, 14, 16). Seven pre-trained networks are provided.
The evaluation results measured by Renyi Entropy for the real-life bat echolocation signal are shown in the following table: 
<table>
<tr>
  <td align="left">SNR</td>
  <td align="center">N=4</td>
  <td align="center">N=6</td>
  <td align="center">N=8</td>
  <td align="center">N=10</td>
  <td align="center">N=12</td>
  <td align="center">N=14</td>
  <td align="center">N=16</td>
</tr>
<tr>
  <td align="left">45 dB</td>
  <td align="center">10.67</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.86</td>
  <td align="center">9.74</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">10.67</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.86</td>
  <td align="center">9.74</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">10.64</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.87</td>
  <td align="center">9.75</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">10.50</td>
  <td align="center">10.02</td>
  <td align="center">9.67</td>
  <td align="center">9.66</td>
  <td align="center">9.66</td>
  <td align="center">9.92</td>
  <td align="center">9.78</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">10.48</td>
  <td align="center">10.27</td>
  <td align="center">9.96</td>
  <td align="center">10.17</td>
  <td align="center">10.11</td>
  <td align="center">10.21</td>
  <td align="center">10.10</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">11.11</td>
  <td align="center">10.77</td>
  <td align="center">10.47</td>
  <td align="center">10.90</td>
  <td align="center">10.72</td>
  <td align="center">10.75</td>
  <td align="center">10.55</td>
</tr>
</table>
It is noted that the network with N=12 has the best performance on the real-life data, which is different from the result on the synthetic data. The reason behind this issue is that overfitting give rise to while increasing N. Thus, for reducing parameters and obtaining great performance, we choose to set N=10.
The visualized experimental results are supplemented as follows:  

![](https://github.com/teki97/KL-TFD/blob/main/supplement.png)


## Contributing Guideline
We would like to thank the authors in these works [2-4] for sharing the source codes of these methods, which are publicly released at https://github.com/Prof-Boualem-Boashash/TFSAP-7.1-software-package, https://github.com/mokhtarmohammadi/Locally-Optimized-ADTFD and https://github.com/Jongchan/attention-module.
## Reference
[1] Jiang, Lei, et al. "Kernel Learning for High-Resolution Time-Frequency Distribution." arXiv preprint arXiv:2007.00322 (2020).  
[2] Park, Jongchan, et al. "Bam: Bottleneck attention module." arXiv preprint arXiv:1807.06514 (2018).  
[3] Boashash, Boualem, and Samir Ouelha. "Designing high-resolution time–frequency and time–scale distributions for the analysis and classification of non-stationary signals: a tutorial review with a comparison of features performance." Digital Signal Processing 77 (2018): 120-152.  
[4] Mohammadi, Mokhtar, et al. "Locally optimized adaptive directional time–frequency distributions." Circuits, Systems, and Signal Processing 37.8 (2018): 3154-3174.  
## Contact
This repo is currently maintained by Lei Jiang (teki97@whu.edu.cn).
