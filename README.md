# Pathmethy

## Introduction of BPformer

DNA methylation, a key epigenetic marker, undergoes significant changes in cancer, making the identification of metastatic tumor origin critical for treatment guidance. Despite advanced diagnostics, 3-9\% of cases remain classified as cancer of unknown primary (CUP). To address this challenge, we present PathMethy, a novel deep learning approach that integrates the Transformer model with biological pathways, including pathway functional categories and the pathway crosstalk network, to analyze DNA methylation in tissue and liquid biopsies. Outperforming seven other methods in accuracy, precision, recall and F1 score across multiple cancer datasets, PathMethy excels in identifying primary and metastatic tumor origins and shows high concordance with previously diagnosed tumor sites in CUP samples. Beyond CUP, PathMethy's clinical utility extends to the identification of primary cancer sites by cell-free DNA methylation (cfDNA) and the classification of molecular subtypes within 11 tumor categories. It provides biological insights by highlighting key pathways, functional categories and their interactions. For broader accessibility, a user-friendly web server for researchers and clinicians is available at https://cup.bpformer.com/index/.

## Installation
Create a conda virtual environment and activate it.

```sh
conda env create -f environment.yml
git clone https://github.com/xmuyulab/Pathmethy.git
```

