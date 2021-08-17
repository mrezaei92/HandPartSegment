# **Weakly-supervised hand part segmentation from depth images**
This repository contains the code for the method introduced in:  

[**Weakly-supervised hand part segmentation from depth images**](https://dl.acm.org/doi/10.1145/3453892.3453902)



#### Qualitative Results
![Architecture sketch for the introduced system](./data/qualitative.png) 


### Usage
1. Download the PyTorch implementation of the Mano hand model from [here](https://github.com/hassony2/manopth)
2. Replace the original manolayer.py with manolayer.py under the folder src (some changes made by us)
3. Run train.py

## Requirements
To run the code you can, e.g., install the following requirements:

 * python 3.
 * [PyTorch](http://pytorch.org/)

## Bibtex
If you use this paper for your research or projects, please cite [Weakly-supervised hand part segmentation from depth images](https://dl.acm.org/doi/10.1145/3453892.3453902).

```bibtex
@inproceedings{rezaei2021weakly,
  title={Weakly-supervised hand part segmentation from depth images},
  author={Rezaei, Mohammad and Farahanipad, Farnaz and Dillhoff, Alex and Elmasri, Ramez and Athitsos, Vassilis},
  booktitle={The 14th PErvasive Technologies Related to Assistive Environments Conference},
  pages={218--225},
  year={2021}
}
