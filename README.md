# Pathmethy

## Introduction of BPformer

Despite advanced diagnostics, 3-5% of cases remain classified as cancer of unknown primary (CUP). DNA methylation, an important epigenetic feature, is essential for determining the origin of metastatic tumors. To address this challenge,  we present PathMethy, a novel deep learning approach that integrates the Transformer model with pathway functional categories and the pathway crosstalk network, to analyze DNA methylation in tissue and liquid biopsies. PathMethy outperformed seven competing methods in F1-score across nine cancer datasets and predicted accurately the molecular subtypes within 9 primary tumor types. It not only excelled at tracing the origins of both primary and metastatic tumors, but also demonstrated a high degree of agreement with previously diagnosed sites in cases of CUP. PathMethy provided biological insights by highlighting key pathways, functional categories and their interactions. Using functional categories of pathways, we gained a global understanding of biological processes. Finally, the clinical utility of PathMethy in non-invasive cancer diagnostics has been validated based on cell-free DNA methylation (cfDNA).  For broader access, a user-friendly web server for researchers and clinicians is available at https://cup.bpformer.com/index/.

## Installation
Create a conda virtual environment and activate it.

```sh
git clone https://github.com/xmuyulab/Pathmethy.git
conda env create -f environment.yml
```

run example:

```python
python3 main.py ./configs/example.yaml    
```


