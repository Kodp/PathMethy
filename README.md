# Overview of PathMethy

## PathMethy: an interpretable AI framework for cancer origin tracing based on DNA methylation

![img](data/img/image.png)

Despite advanced diagnostics, 3-5% of cases remain classified as cancer of unknown primary (CUP). DNA methylation, an important epigenetic feature, is essential for determining the origin of metastatic tumors. We presented PathMethy, a novel Transformer model integrated with functional categories and crosstalk of pathways, to accurately tracing the origin of tumors in CUPs based on DNA methylation. PathMethy outperformed seven competing methods in F1-score across nine cancer datasets and predicted accurately the molecular subtypes within 9 primary tumor types. It not only excelled at tracing the origins of both primary and metastatic tumors, but also demonstrated a high degree of agreement with previously diagnosed sites in cases of CUP. PathMethy provided biological insights by highlighting key pathways, functional categories and their interactions. Using functional categories of pathways, we gained a global understanding of biological processes. For broader access, a user-friendly web server for researchers and clinicians is available at https://cup.pathmethy.com.

# 🚀 Usage

## Method 1: Using conda and git

Clone the repository or download the ZIP file of the repository. The repository contains the model and data, which are somewhat large. Downloading the ZIP file might be faster. Cloning the repository takes about 10 minutes.

```sh
git clone https://github.com/Kodp/PathMethy.git  # clone the repo
cd PathMethy
conda create -n PathMethy python=3.10 # Combine weights and data files
conda activate PathMethy
pip install --requirement requirements.txt
python ./utils/file_chunk.py combine ./metadata.yaml  
```

> This conda environment is using GPU-accelerated PyTorch. To use this environment, the CUDA version must be 12.0 or higher.

**Run:**

```python
python3 main.py ./configs/example.yaml  
```

The YAML file contains the experiment configuration. You can customize your experiment by using this config. Check the comments in `example.yaml` to know how to do it.

Each experiment's files are located in the `experiments` directory. The name of each subdirectory is the experiment name, which is configured in the YAML file under the `.exp_name` key.

## Method 2: Using Docker

We' ve packaged the code, model weights, and dependencies into a single Docker container.

**Get and start the image:**

```shell
docker pull 000199/pathmethy:latest
docker run -it --name PathMethy_container 000199/pathmethy:latest
```

> This Docker is configured to use the CPU only, so there's no need for CUDA or a graphics card.

**Run:**

```shell
cd PathMethy/
python3 main.py ./configs/example.yaml  
```

## Contact

If you encounter any problems, bugs, or have any questions, feel free to submit an issue. Alternatively, please contact [songyh@stu.xmu.edu.cn](mailto:cying2023@stu.xmu.edu.cn) or [xiejiajing@stu.xmu.edu.cn](mailto:xiejiajing@stu.xmu.edu.cn).
