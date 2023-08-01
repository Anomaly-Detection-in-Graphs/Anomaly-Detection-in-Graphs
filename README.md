
## Implementing environment:  
- accelerate==0.17.1
- dgl==1.1.0
- matplotlib==3.5.0
- networkx==2.8.4
- numpy==1.21.4
- ogb==1.3.5
- pandas==2.0.0
- pyvis==0.3.2
- PyYAML==6.0
- PyYAML==6.0.1
- scikit_learn==1.1.3
- scipy==1.11.1
- torch==1.10.0+cu113
- torch_geometric==2.2.0
- torch_scatter==2.1.1
- torch_sparse==0.6.13
- tqdm==4.61.2

- GPU: Tesla V100 32G

## Demo on Training and Evalation:  
```bash
python GCL_processed_mini_batch_ver_2.py
```

## Fraud Detection AUC
The best test AUC is 82.309 with resgatedgraphconvolution, feature engineering, GCL and mini-batch sampling.
