# Pathmethy

## Introduction of BPformer

Despite advanced diagnostics, 3-5% of cases remain classified as cancer of unknown primary (CUP). DNA methylation, an important epigenetic feature, is essential for determining the origin of metastatic tumors. To address this challenge,  we present PathMethy, a novel deep learning approach that integrates the Transformer model with pathway functional categories and the pathway crosstalk network, to analyze DNA methylation in tissue and liquid biopsies. PathMethy outperformed seven competing methods in F1-score across nine cancer datasets and predicted accurately the molecular subtypes within 9 primary tumor types. It not only excelled at tracing the origins of both primary and metastatic tumors, but also demonstrated a high degree of agreement with previously diagnosed sites in cases of CUP. PathMethy provided biological insights by highlighting key pathways, functional categories and their interactions. Using functional categories of pathways, we gained a global understanding of biological processes. Finally, the clinical utility of PathMethy in non-invasive cancer diagnostics has been validated based on cell-free DNA methylation (cfDNA).  For broader access, a user-friendly web server for researchers and clinicians is available at https://cup.bpformer.com/index/.

Here's a more visually appealing version of your README for GitHub, with added emojis and Markdown formatting enhancements to make it more engaging:

---

# 🚀 Installation

```sh
git clone https://github.com/xmuyulab/Pathmethy.git
conda create -n PathMethy python=3.10
conda activate PathMethy
pip install --requirement requirements.txt
```

The model is too large to be directly uploaded to GitHub, so we use Git LFS for the upload. You can use two methods to obtain the model and test data:

**Git LFS**

🔗 Install from the official website: [Git Large File Storage](https://git-lfs.github.com/)

or

```sh
sudo apt-get install git-lfs  
```

After installing Git LFS and cloning the repository, run:

```sh
git lfs pull
```

This command will automatically download the model and test data. It will download three files:

```
MODEL/model.pth.tar
data/cor_matrix50.csv
data/test_data.pkl
```

> Hint. If you just git clone the repo, the above three files will only generate placeholders, not the files themselves.

**Baidu**

link: https://pan.baidu.com/s/1WVSmzfc0M8U_AVhPO3ZhNA?pwd=t54k
code: t54k

## 🏃‍♂️ Run

**Example run:**

```python
python3 main.py ./configs/example.yaml  
```

The YAML file contains the experiment configuration. You can customize your experiment by using this config. Check the comments in `example.yaml` to know how to do it.

Each experiment's files are located in the `experiments` directory. The name of each subdirectory is the experiment name, which is configured in the YAML file under the `.exp_name` key.
