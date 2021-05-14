# Deep Multi-Type Knowledge Tracing
Code for our paper:

```
C. Wang, S. Zhao, and S. Sahebi. Learning from Non-Assessed Resources: Deep Multi-TypeKnowledge Tracing. In Proceedings of The 14th International Conference onEducational Data Mining, 2021.
```


## How to install and run 

For example, to run the DKT model:
```angular2html
cd DMKT 
pip install -e .
cd dmkt_exp
python run.py -c configs/dmkt_exp_0.json
```

Another way to run using conda:
```angular2html
cd DMKT 
conda env create -f environment.yml
source init_env.sh
cd dmkt 
python run.py -c configs/dmkt_exp_0.json
```


## Cite:

Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2021dmkt,
  title={Learning from Non-Assessed Resources: Deep Multi-Type Knowledge Tracing},
  author={Wang, Chunpai and Zhao, Siqian and Sahebi,Shaghayegh},
  booktitle={Proceedings of the 14th International Conference on Educational Data Mining (EDM-2021)},
  year={2021}
}
```

## Acknowledgement:

This  paper  is  based  upon  work  sup-ported by the National Science Foundation under Grant No.1755910
