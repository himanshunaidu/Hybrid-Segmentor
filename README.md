## 	Hybrid-Segmentor: Hybrid Approach for Automated Fine-Grained Crack Segmentation in Civil Infrastructure - Automation in Construction [IF:11.5]
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybrid-segmentor-a-hybrid-approach-to/crack-segmentation-on-crackvision12k)](https://paperswithcode.com/sota/crack-segmentation-on-crackvision12k?p=hybrid-segmentor-a-hybrid-approach-to)

[![Static Badge](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2409.02866) &nbsp;
[![Static Badge](https://img.shields.io/badge/Dataset-CrackVision12K-green)](https://rdr.ucl.ac.uk/articles/dataset/CrackVision12K/26946472?file=49023628) &nbsp;

## 1. Model Architecture
![](./figures/model_architecture.png)

**Hybrid-Segmentor model weight**: 

1. Best Model: [Best model weight](https://1drv.ms/u/s!AtFigR8so_Ssr74icViwvNvnpzdbVg?e=JB4dEO)
2. All weights: [models weight folder](https://1drv.ms/u/s!AtFigR8so_Ssr74kDbpbEjUGqx1Z3A?e=uhRgPu)
   
**All weights Folder Structure:**
```
model_weights.zip
  │
  ├── ablation_encoder
  │     ├── transformer_path.ckpt (Trained only transformer path)
  │     └── CNN_path.ckpt         (Trained only CNN path)
  │
  └── ablation_loss
        ├── hybrid_segmentor_DICE.ckpt    (Dice Loss) -> λ = 0
        ├── hybrid_segmentor_BCE_1.ckpt   (BCE-DICE loss with λ = 0.1 ) 
        ├── hybrid_segmentor_BCE_2.ckpt   (BCE-DICE loss with λ = 0.2 )
        ├── hybrid_segmentor_BCE_3.ckpt   (BCE-DICE loss with λ = 0.3 )
        ├── hybrid_segmentor_BCE_4.ckpt   (BCE-DICE loss with λ = 0.4 )
        ├── hybrid_segmentor_BCE_5.ckpt   (BCE-DICE loss with λ = 0.5 )
        ├── hybrid_segmentor_BCE_6.ckpt   (BCE-DICE loss with λ = 0.6 )
        ├── hybrid_segmentor_BCE_7.ckpt   (BCE-DICE loss with λ = 0.7 )
        ├── hybrid_segmentor_BCE_8.ckpt   (BCE-DICE loss with λ = 0.8 )
        ├── hybrid_segmentor_BCE_9.ckpt   (BCE-DICE loss with λ = 0.9 )
        ├── hybrid_segmentor_BCE.ckpt     (BCE loss) -> λ = 1
        └── hybrid_segmentor_recall.ckpt  (Recall Loss)
```
#### If you use our model in your research, please cite "Hybrid-Segmentor Reference" below.

## 2. Refined Dataset (CrackVision12K)
The refined dataset is developed with 13 publicly available datasets that have been refined using image processing techniques.
**Please note that the use of our dataset is RESTRICTED to non-commercial research and educational purposes.**

**Dataset**: [CrackVision12K](https://rdr.ucl.ac.uk/articles/dataset/CrackVision12K/26946472?file=49023628).
|Folder|Sub-Folder|Description|
|:----|:-----|:-----|
|`train`|IMG / GT|RGB images and binary annotation for training|
|`test`|IMG / GT|RGB images and binary annotation for testing|
|`val`|IMG / GT|RGB images and binary annotation for validation|

#### To download the dataset from the link, please cite "Hybrid-Segmentor & CrackVision12K" below.

## 3. Set-Up
1. Pytorch 2.5.0 + CUDA 11.8
2. Pytorch Lightning 2.4.0

**Training**
Before training, change variables such as dataset path, batch size, etc in config.py. 
```
cd Hybrid_Segmentor
python trainer.py
```

**Testing**
Before testing, change the model name and output folder path.
```
cd Hybrid_Segmentor
python test.py
```
## 4. Results
![](./figures/figure_5.png)
Example crack images segmented by our model and benchmarked models. The red ovals highlight the areas where our model outperforms other benchmarked models. In examples without red ovals, such as (F) and (H), our model demonstrates strong performance across overall structures.

## Citaitons
 - **Hybrid-Segmentor & CrackVision12K Reference**:
   
   ***If you use our model or dataset, please cite the following***:
```
@article{GOO2025105960,
      title = {Hybrid-Segmentor: Hybrid approach for automated fine-grained crack segmentation in civil infrastructure},
      journal = {Automation in Construction},
      volume = {170},
      pages = {105960},
      year = {2025},
      issn = {0926-5805},
      doi = {https://doi.org/10.1016/j.autcon.2024.105960},
      url = {https://www.sciencedirect.com/science/article/pii/S0926580524006964},
      author = {June Moh Goo and Xenios Milidonis and Alessandro Artusi and Jan Boehm and Carlo Ciliberto},
      keywords = {Deep learning applications, Semantic segmentation, Convolutional neural networks, Transformers, Hybrid approach, Crack detection, Crack dataset, Fine-grained details},
}
@misc{goo2024hybridsegmentor,
      title={Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructure}, 
      author={June Moh Goo and Xenios Milidonis and Alessandro Artusi and Jan Boehm and Carlo Ciliberto},
      year={2024},
      eprint={2409.02866},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02866}, 
}
```

---
 - **Sub-Dataset Reference**:
1. Aigle-RN / ESAR / LCMS Datasets [Dataset Link](https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html)
```
@article{AEL_dataset,
  title={Automatic crack detection on two-dimensional pavement images: An algorithm based on minimal path selection},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={10},
  pages={2718--2729},
  year={2016},
  publisher={IEEE}
}
```
2. SDNet2018 Datasets [Dataset Link](https://digitalcommons.usu.edu/all_datasets/48/)
```
@article{sdnet2018,
  title={SDNET2018: A concrete crack image dataset for machine learning applications},
  author={Maguire, Marc and Dorafshan, Sattar and Thomas, Robert J},
  year={2018},
  publisher={Utah State University}
}
```
3. Masonry Datasets [Dataset Link](https://github.com/dimitrisdais/crack_detection_CNN_masonry)
```
@article{masonry_dataset,
  author = {Dais, Dimitris and Bal, Ihsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},
  doi = {10.1016/j.autcon.2021.103606},
  journal = {Automation in Construction},
  pages = {103606},
  title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},
  volume = {125},
  year = {2021}
}
```
4. Crack500 Dataset [Dataset Link](https://github.com/fyangneil/pavement-crack-detection)
```
@inproceedings{crack500_dataset,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={2016 IEEE international conference on image processing (ICIP)},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}
```
5. CrackLS315 / CRKWH100 / CrackTree260 / Stone331 Datasets [Github Link](https://github.com/qinnzou/DeepCrack) [Direct Link-passcodes: zfoo](https://pan.baidu.com/s/1PWiBzoJlc8qC8ffZu2Vb8w)
```
@article{Deep_crack_crackLS315,
  title={Deepcrack: Learning Hierarchical Convolutional Features for Crack Detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}
```
6. DeepCrack Dataset [Dataset Link](https://github.com/yhlleo/DeepCrack)
```
@article{deepcrack_dataset,
title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
journal={Neurocomputing},
volume={338},
pages={139--153},
year={2019},
doi={10.1016/j.neucom.2019.01.036}
}
```
7.1 GAPS384 7.2 GAPs (Original Dataset and paper) [GAPS384 Dataset Link](https://github.com/fyangneil/pavement-crack-detection) [GAPs Dataset Link](https://www.tu-ilmenau.de/neurob/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)
```
@article{FPHBN_gaps384,
title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
journal={IEEE Transactions on Intelligent Transportation Systems}, year={2019}, publisher={IEEE} }

@inproceedings{GAPS_data_original,
title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike and Gross, Horst-Michael},
booktitle={International Joint Conference on Neural Networks (IJCNN)}, pages={2039--2047}, year={2017} }
```
8. CFD Dataset [Dataset Link](https://github.com/cuilimeng/CrackForest-dataset)
```
@article{CFD1,
title={Automatic road crack detection using random structured forests},
author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
journal={IEEE Transactions on Intelligent Transportation Systems},volume={17},number={12},
pages={3434--3445},year={2016},publisher={IEEE}}

@inproceedings{CFD2,
title={Pavement Distress Detection Using Random Decision Forests},
author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},
booktitle={International Conference on Data Science},
pages={95--102},
year={2015},
organization={Springer}
}
```

If you have any questions, please contact me: june.goo.21 @ ucl.ac.uk without hesitation.
