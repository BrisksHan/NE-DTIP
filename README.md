# NE-DTIP
[A Network Embedding Based Approach to Drug-Target Interaction Prediction Using Additional Implicit Networks, ICANN2021, to appear](https://github.com/BrisksHan/NE-DTIP/blob/main/NE-DTIP_ICANN%202021.pdf)

```
@inproceedings{zhang2021network,
  title={A Network Embedding Based Approach to Drug-Target Interaction Prediction Using Additional Implicit Networks},
  author={Zhang, Han and Hou, Chengbin and McDonald, David and He, Shan},
  booktitle={International Conference on Artificial Neural Networks},
  year={2021},
  organization={Springer}
}
```

## Brief Introduction
Unlike previous DTI prediction methods, the proposed method additionally considers two homogeneous networks, i.e., TIN and DIN, both of which are generated based on the implicit relations of a given DTI network.
<center>
    <img src="https://github.com/BrisksHan/NE-DTIP/blob/main/data/Fig6.1.PNG" width="500"/>
</center>

The overview of the proposed method, which includes two stages: the feature vector construction stage and the DTI classification stage.
<center>
    <img src="https://github.com/BrisksHan/NE-DTIP/blob/main/data/Fig6.2.PNG" width="666"/>
</center>

Apart from the benchmark over several datasets and state-of-the-art methods, we also conduct a case study to verify the top-20 DTI predictions by the proposed method on DT-IN dataset.
<center>
    <img src="https://github.com/BrisksHan/NE-DTIP/blob/main/data/Fig6.3.JPG" width="800"/>
</center>
The top-100 DTIs predicted by the proposed method on DT-IN dataset. Each circle represents a drug. Each diamond represents a target. The edges between nodes indicate novel DTI predictions (i.e., new DTIs not recorded in the dataset). The top-20 DTI predictions are in solid lines, while the remaining ones are in dashed lines. <br>
We search for the supporting studies over the top-20 predicted DTIs. It is interesting to find that six of them are supported by recent studies. Sunitinib inhibits EPHB2 \cite{martinho2013vitro}, GSK3B \cite{calero2014sunitinib}, and SYK \cite{noe2016clinical}. The treatment of using Sunitinib substantially increases ErbB3 \cite{harvey2015oestrogen}. Bosutinib is an inhibitors to KDR \cite{brown2017cardiovascular}. Haloperidol would down regulate CHRM2 \cite{swathy2017haloperidol}.


## Install and Usage
python version: 3.6

package require: numpy scipy gensim networkx

the version will be added soon.

The approach take DTI network, Drug Strucutural Similarity Network and Taget Sequence Similarity Network as input.

There are demo provided. The input should be pkl file of networkx object. 

Later, we will add a new function to take .txt file as input.

If you have any questions, please send a Email to hanzhang89@qq.com
