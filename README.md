# Multi-Class Text Classification with Transformers

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN



### Getting started


- Install PyTorch and other dependencies.

  please run the command `pip install -r requirements.txt`.

### Training and Evaluation
- Go to the `Scripts` directory:
```bash
cd Scripts
```

- Train stand-alone model (BERT-base-uncased) with GPU support:
```bash
python train.py
```
- Train stand-alone model (BERT-base-uncased) with CPU only:
```bash
python train.py --device CPU
```
- Train stand-alone model with Twitter-parsed dataset:
```bash
python train.py --dataset Twitter
```
Information regarding other training parameters can be found at `Scripts/common.py` file.

Fine-tuned models will be saved at `../Models/` folder.
Evaluation output files will be saved at `../Output/` folder.
Figures will be saved at `../Figures/` folder.


### Citation
If you use this code for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9671594/).
```
@inproceedings{ahmed2021being,
  title={Am I Being Bullied on Social Media? An Ensemble Approach to Categorize Cyberbullying},
  author={Ahmed, Tasnim and Kabir, Mohsinul and Ivan, Shahriar and Mahmud, Hasan and Hasan, Kamrul},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},
  pages={2442--2453},
  year={2021},
  organization={IEEE}
}

```

 
