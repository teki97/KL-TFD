# Kernel Learning for High-Resolution Time-frequency Distribution
We provide a pytorch implementation of the paper: Kernel Learning for High-Resolution Time-frequency Distribution [1], where a kernel learning based TFD model is proposed to gain high resolution and CT-free TFDs. The proposed model includes two kinds of multi-channel learning covolutional kernels stacked, that is, normal 2D Conv block and Skipping 2D Conv Block. Specifically, the former has large kernel size so that a smooth TFD can be attained while the latter has small kernel size with BAM [2] to improve resolution. Five pre-trained networks are provided, and they are corresponding to five cases of various K (6, 7, 8, 9, 10) in this paper respectively.

## Preparation
- python 3.6
- pytorch 0.4.1
- cuda 9.0
- cudNN 7.6.3.30

## Supplementary
In this paper, we discuss the robustness of our proposed method, i.e., we have some experiments with the increase of K, and it is examined that the performance can be improved by increasing K. 
The evaluation results measured by l1 distance for the first synthetic signal (two-component synthetic signal in our paper) are shown in the following table: 
<table>
<tr>
  <td align="left">SNR</td>
  <td align="center">K=6</td>
  <td align="center">K=7</td>
  <td align="center">K=8</td>
  <td align="center">K=9</td>
  <td align="center">K=10</td>
</tr>
<tr>
   <td align="left">45 dB</td>
  <td align="center">1.49</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.32</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">1.49</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.32</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">1.50</td>
  <td align="center">1.43</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.33</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">1.53</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.37</td>
  <td align="center">1.35</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">1.64</td>
  <td align="center">1.61</td>
  <td align="center">1.53</td>
  <td align="center">1.53</td>
  <td align="center">1.50</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">1.78</td>
  <td align="center">1.77</td>
  <td align="center">1.66</td>
  <td align="center">1.62</td>
  <td align="center">1.61</td>
</tr>
</table>

The visualized experimental results are supplemented as follows:  
![](https://github.com/teki97/kernel-learning-time-frequency-distribution/blob/main/supplemet_1.jpg)

## Contributing Guideline
We would like to thank the authors in these works [2-4] for sharing the source codes of these methods, which are publicly released at https://github.com/Prof-Boualem-Boashash/TFSAP-7.1-software-package, https://github.com/mokhtarmohammadi/Locally-Optimized-ADTFD and https://github.com/Jongchan/attention-module.
## Reference
[1] Jiang, Lei, et al. "Kernel Learning for High-Resolution Time-Frequency Distribution." arXiv preprint arXiv:2007.00322 (2020).  
[2] Park, Jongchan, et al. "Bam: Bottleneck attention module." arXiv preprint arXiv:1807.06514 (2018).  
[3] Boashash, Boualem, and Samir Ouelha. "Designing high-resolution time–frequency and time–scale distributions for the analysis and classification of non-stationary signals: a tutorial review with a comparison of features performance." Digital Signal Processing 77 (2018): 120-152.  
[4] Mohammadi, Mokhtar, et al. "Locally optimized adaptive directional time–frequency distributions." Circuits, Systems, and Signal Processing 37.8 (2018): 3154-3174.  
## Contact
This repo is currently maintained by Lei Jiang (teki97@whu.edu.cn).
