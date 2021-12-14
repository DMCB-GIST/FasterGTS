# FasterGTS
## The overview
A faster molecular generative model with genetic algorithm and tree search for cancer samples
![image](https://user-images.githubusercontent.com/31497898/145931441-5c5dd07f-ab61-4b1c-8e56-b3087e29716c.png)
Figure 1. The overall workflow of FasterGTS for generating cancer sample-specific drugs.
![image](https://user-images.githubusercontent.com/31497898/145931614-9f2705e5-b899-4273-853d-fe06a38e43d4.png)
Figure. 2. The workflow of MCTS in FasterGTS at st.


## Requirements
pytorch >= 1.6.0

pip install anytree

pip install scaffoldgraph

conda install -c rdkit rdkit

pip install fcd

install https://github.com/connorcoley/scscore

conda install -c conda-forge tqdm

conda install -c rdkit -c mordred-descriptor mordred

hickle >= 2.1.0

TensorFlow==1.13.1

Keras==2.1.4
