# [TIP 2024] Official Implementation of Progressive Adaptive Multimodal Fusion Network (PAMFN)


![](./resources/framework.jpg)





# Installation
## Build the python environment
Codes are tested on RTX 3090 and I not sure you can get the same results on different GPUs or different python environment.
```
1. conda create -n PAMFN python=3.8 -y
2. conda activate PAMFN
3. conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
4. pip install -r requires.txt
```

## Download extracted features and pretrained models
The extracted features and pretrained models can be downloaded from [here](https://1drv.ms/u/s!ApyE_Lf3PFl2itlpIvyjNzcQ0TlsPA?e=biQ4bw) and should be placed in the current directory.
```
./
├── data/
└── pretrained_models
```

# Evaluation
Using the follow command to evaluate the pretrained model:
```python
python main.py --gpu {gpu_id} --feats {feature_type} --action {action_type} --multi_modality --test
```
- {gpu_id}: The GPU device ID. 
- {feature_type}: Set as 2 to use the features extracted by UNMT, I3D, and MAST. Set as 1 to use the features extracted by VST, I3D, and AST.
- {action_type}: Ball, Clubs, Hoop, Ribbon, TES, PCS.

# Training 
## Training the modality-specific branch
Using the follow command to train a modality-specific branch:
```
python main.py --gpu {gpu_id} --feats {feature_type} --action {action_type} --modality {modality_type}
```
- {gpu_id}: The GPU device ID. 
- {feature_type}: Set as 2 to use the features extracted by UNMT, I3D, and MAST. Set as 1 to use the features extracted by VST, I3D, and AST.
- {action_type}: Ball, Clubs, Hoop, Ribbon, TES, PCS.
- {modality_type}: Set as V/F/A to use RGB/Optical flow/Audio features.

An Example:
```
python main.py --gpu 0 --feats 2 --action Ball --modality V
```
## Training the mixed-modality branch
Using the follow command to train the mixed-modality branch:
```
python main.py --gpu {gpu_id} --feats {feature_type} --action {action_type} --multi_modality
```
An Example:
```
python main.py --gpu 0 --feats 2 --action Ball --multi_modality
```

# Citation
Please cite this work if you find it useful:
```
@article{zeng2024multimodal,
  title={Multimodal Action Quality Assessment},
  author={Zeng, Ling-An and Zheng, Wei-Shi},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
