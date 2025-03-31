# TCGAN: Temporal Convolutional Generative Adversarial Network for Fetal ECG Extraction Using Single-channel Abdominal ECG

⭐ our [article](https://ieeexplore.ieee.org/document/10818591) ⭐ 
⭐ The main contribution of our work is that we design a temporal convolutional generative adversarial network to extract FECG signal using single-channel AECG signal ⭐ 

<iframe src="https://docs.google.com/viewer?url=https://github.com/zhwt-xidian/TCGAN/raw/master/img/fig1.pdf&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>

<p align="center"> <img src="Fig/framework.png" width="60%"> </p>


## Requirements

```python
pip install -r requirements.txt
```
## File Description
### data
This includes preprocessing of the data and preparation of the dataset.
### Dataset
It contains a classic non-invasive fetal ECG dataset: ADFECGDB.
### options
This mainly includes training and testing parameters, such as learning rate, epochs and so on. If you want to adjust the training status, you should see this file.
### Model.py
This is the code for the model: TCGAN.
### Common_blocks
It contains the common blocks which are used in TCGAN.
### Result
It includes the results of Train and Test.


## Train
```
python train.py
```
## Results
这里放Fig.2

## Citation
If our code is helpful to you, please cite:

```
@article{r1,
  title={ TCGAN: Temporal Convolutional Generative Adversarial Network for Fetal ECG Extraction Using Single-channel Abdominal ECG},
  author={Zhen -Zhen Huang, Wei -Tao Zhang, Yang Li, Jian Cui and Ya -Ru Zhang},
  journal={ IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

