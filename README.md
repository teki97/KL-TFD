# Kernel Learning for High-Resolution Time-frequency Distribution
![](https://github.com/teki97/kernel-learning-time-frequency-distribution/blob/main/supplement.png)
We provide a pytorch implementation of the paper: Kernel Learning for High-Resolution Time-Frequency Distribution [1], where a kernel learning based time-frequency distribution (TFD) model is proposed to gain high resolution and CT-free TFDs. As shown in the above figure, the proposed model includes **P** normal 2D Conv block and **Q** Skipping 2D Conv Block. Specifically, the former has large kernel size so that a smooth TFD can be attained while the latter has small kernel size with bottleneck attention module (BAM) [2] to improve resolution. 

## Preparation
- python 3.6
- pytorch 0.4.1
- cuda 9.0
- cudNN 7.6.3.30

## Supplementary

### Discussion on Q
In this paper, we discuss the robustness of our proposed method, i.e., we have some experiments with the increase of Q, and it is examined that the performance can be improved by increasing Q. Six pre-trained networks are provided (training signals at SNR = **10 dB**), and they are corresponding to six cases of various Q (0, 1, 2, 3, 4, 5).
The evaluation results measured by l1 distance for the two-component synthetic signal are shown in the following table: 
<table>
<tr>
  <td align="left">SNR</td>
  <td align="center">Q=0</td>
  <td align="center">Q=1</td>
  <td align="center">Q=2</td>
  <td align="center">Q=3</td>
  <td align="center">Q=4</td>
  <td align="center">Q=5</td>
</tr>
<tr>
   <td align="left">45 dB</td>
  <td align="center">3.05</td>
  <td align="center">1.49</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.32</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">3.05</td>
  <td align="center">1.49</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.32</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">3.05</td>
  <td align="center">1.50</td>
  <td align="center">1.43</td>
  <td align="center">1.40</td>
  <td align="center">1.36</td>
  <td align="center">1.33</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">3.04</td>
  <td align="center">1.53</td>
  <td align="center">1.44</td>
  <td align="center">1.40</td>
  <td align="center">1.37</td>
  <td align="center">1.35</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">3.19</td>
  <td align="center">1.64</td>
  <td align="center">1.61</td>
  <td align="center">1.53</td>
  <td align="center">1.53</td>
  <td align="center">1.50</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">3.70</td>
  <td align="center">1.78</td>
  <td align="center">1.77</td>
  <td align="center">1.66</td>
  <td align="center">1.62</td>
  <td align="center">1.61</td>
</tr>
</table>

The visualized experimental results are supplemented as follows:  
![](https://github.com/teki97/kernel-learning-time-frequency-distribution/blob/main/supplemet_1.jpg)



### Discussion on P

### Discussion on training data
Then, when we train our network using signals at SNR = 5 dB, and it is examined that performance with low SNR can be improved while weak energy parts are prone to be ignored. The evaluation results measured by l1 distance for the three-component synthetic signal and the evaluation results measured by Renyi Entropy for the real-world bat echolocation signal are shown in the following tables:
<table>
  <tr>
    <td>
    <table>
<tr>
  <td align="left">SNR</td>
  <td align="center">Q=0</td>
  <td align="center">Q=1</td>
  <td align="center">Q=2</td>
  <td align="center">Q=3</td>
  <td align="center">Q=4</td>
  <td align="center">Q=5</td>
</tr>
<tr>
   <td align="left">45 dB</td>
  <td align="center">3.75</td>
  <td align="center">1.38</td>
  <td align="center">1.22</td>
  <td align="center">1.15</td>
  <td align="center">1.08</td>
  <td align="center">1.03</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">3.75</td>
  <td align="center">1.38</td>
  <td align="center">1.22</td>
  <td align="center">1.15</td>
  <td align="center">1.08</td>
  <td align="center">1.03</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">3.75</td>
  <td align="center">1.38</td>
  <td align="center">1.22</td>
  <td align="center">1.15</td>
  <td align="center">1.09</td>
  <td align="center">1.03</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">3.76</td>
  <td align="center">1.41</td>
  <td align="center">1.25</td>
  <td align="center">1.19</td>
  <td align="center">1.11</td>
  <td align="center">1.05</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">3.91</td>
  <td align="center">1.54</td>
  <td align="center">1.43</td>
  <td align="center">1.35</td>
  <td align="center">1.25</td>
  <td align="center">1.23</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">4.37</td>
  <td align="center">1.62</td>
  <td align="center">1.58</td>
  <td align="center">1.52</td>
  <td align="center">1.45</td>
  <td align="center">1.49</td>
</tr>
</table>
      </td>
    <td>
          <table>
<tr>
  <td align="left">SNR</td>
  <td align="center">Q=0</td>
  <td align="center">Q=1</td>
  <td align="center">Q=2</td>
  <td align="center">Q=3</td>
  <td align="center">Q=4</td>
  <td align="center">Q=5</td>
</tr>
<tr>
   <td align="left">45 dB</td>
  <td align="center">11.98</td>
  <td align="center">10.71</td>
  <td align="center">10.63</td>
  <td align="center">10.42</td>
  <td align="center">10.17</td>
  <td align="center">10.16</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">11.98</td>
  <td align="center">10.71</td>
  <td align="center">10.63</td>
  <td align="center">10.42</td>
  <td align="center">10.17</td>
  <td align="center">10.16</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">11.98</td>
  <td align="center">10.71</td>
  <td align="center">10.63</td>
  <td align="center">10.41</td>
  <td align="center">10.17</td>
  <td align="center">10.16</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">11.98</td>
  <td align="center">10.70</td>
  <td align="center">10.60</td>
  <td align="center">10.40</td>
  <td align="center">10.16</td>
  <td align="center">10.16</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">12.03</td>
  <td align="center">10.50</td>
  <td align="center">10.36</td>
  <td align="center">10.36</td>
  <td align="center">10.34</td>
  <td align="center">10.44</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">12.30</td>
  <td align="center">10.47</td>
  <td align="center">10.33</td>
  <td align="center">10.48</td>
  <td align="center">10.48</td>
  <td align="center">10.65</td>
</tr>
</table>
      </td>
    </tr>
  </table>
  
The visualized experimental results are supplemented as follows:    
![](https://github.com/teki97/kernel-learning-time-frequency-distribution/blob/main/supplement_2.jpg)

## Contributing Guideline
We would like to thank the authors in these works [2-4] for sharing the source codes of these methods, which are publicly released at https://github.com/Prof-Boualem-Boashash/TFSAP-7.1-software-package, https://github.com/mokhtarmohammadi/Locally-Optimized-ADTFD and https://github.com/Jongchan/attention-module.
## Reference
[1] Jiang, Lei, et al. "Kernel Learning for High-Resolution Time-Frequency Distribution." arXiv preprint arXiv:2007.00322 (2020).  
[2] Park, Jongchan, et al. "Bam: Bottleneck attention module." arXiv preprint arXiv:1807.06514 (2018).  
[3] Boashash, Boualem, and Samir Ouelha. "Designing high-resolution time–frequency and time–scale distributions for the analysis and classification of non-stationary signals: a tutorial review with a comparison of features performance." Digital Signal Processing 77 (2018): 120-152.  
[4] Mohammadi, Mokhtar, et al. "Locally optimized adaptive directional time–frequency distributions." Circuits, Systems, and Signal Processing 37.8 (2018): 3154-3174.  
## Contact
This repo is currently maintained by Lei Jiang (teki97@whu.edu.cn).
