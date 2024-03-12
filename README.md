 DRS-GCN: Counteracting Over-smoothing through Dynamic Reorganization and Smoothness Loss
====


This is a Pytorch implementation of paper: DRS-GCN: Counteracting Over-smoothing through Dynamic Reorganization and Smoothness Loss


## Requirements

  * Python 3.9.7
  * torch==1.12.1+cu116
  * numpy==1.20.3
  * GraphRicciCurvature==0.5.3.1
  * networkx==2.6.3


## Usage
You can change some parameters by edit 
```config.py```

And start it by run this demo
```python main.py```

## Data
The data format is same as [GCN](https://github.com/tkipf/gcn). We provide three benchmark datasets as examples (see `data` folder). We use the public dataset splits provided by [Planetoid](https://github.com/kimiyoung/planetoid). 

## Results
The experimental outcomes generated after execution, comprising accuracy of identification and loss logs, will be printed on the screen. The trained weight files will be stored at the specified path you have configured in the file
```config.py```.
## Change Log
 
## References
```

```



