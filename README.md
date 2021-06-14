# Brain Tumor Diagnosis

## Business Problem

Brain tumor is one of the deadliest forms of cancer and as all cancer types, early detection is very important and can save lives. A growing interest has been seen in deep learning based semantic segmentation. UNet, which is one of deep learning networks with an encoder-decoder architecture, is widely used in medical image segmentation. Combining multi-scale features is one of important factors for accurate segmentation. 

There are three types of brain tumors menigioma, pituitray and glioma. the type of the tumor can be an indication of the tumor‘s aggressiveness. Radiation or surgery treatment for brain tumors. Radiation treatment consists of high-energy radiation source like gamma rays or x-rays shoots in a very precise way at the tumor and therefore kill the tumor‘s cells while sparing surrounding tissues. Doctors need to segment the infected tissues by separate the infected cells from the healthy ones. Creating this segmentation map in an accurate way is very tedious, expensive, time-consuming and error-prone task.

## Solution

Use deep learning to automate the precise detection, identfication, and the segmention process for brain tumor diagnosis.

## Data

Source of the dataset: https://figshare.com/articles/brain_tumor_dataset/1512427

About the dataset: This brain tumor dataset containing 3064 T1 MRIs from 233 patients with three kinds of brain tumor: Meningioma (708 slices), Glioma (1426 slices), Pituitary tumor (930 slices).

This data is organized in matlab data format (.mat file). Each file stores a struct containing the following fields for an image:

cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor

cjdata.PID: patient ID

cjdata.image: image data

cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.We can use it to generate binary image of tumor mask.

cjdata.tumorMask: a binary image with 1s indicating tumor region

In this repository, 3 Notebooks are found:

a) DataTransformation_Exploration.iypnb

b) Brain_Tumor_Classification_Models.iypnb

c) Brain_Tumor_segmentation_models.iypnb


The trained classification and segmentation models are found in 'saved_models' folder. A presentation of major results can be found as pdf file 'Brain Tumor Diagnosis_presentation'


## Methodology

Task 1: Develop models for the detction and identification of brain tumors (PyTorch).

Task 2: Develop deep learning based semantic segmentation for brain tumors (PyTorch).

### Task 1

Instantiate the transfer learning model using torchvision's models class. RESNET50  and VGG16 are the CNN models that we're going to use by transfer learning.

All the pretrained weights are set to  trainable by enabling every layer's parameters as true.  The top layer is built by creating a custom output sequential layer and assigned it to model's fc. Model's loss function is set as CrossEntropyLoss.

Training configuration: 50 epochs, optimizerSGD, and learning rate of 0.0003.

### Task 2

Five deep learning semantic segmentation models are tarined for the brain tumor data set:

1) UNet

2) ResUNet

3) Deep ResUNet

4) ONet

5) Dynamic UNet

Training configuration= 100 epochs, learnin rate=0.001, batch size= 2 images

Evaluation of segmentation models is based on dice coefficient. Dice coefficient is the precentage of true tumor pixel.

![image](https://user-images.githubusercontent.com/53411455/121936946-28105400-cd18-11eb-9372-035770ef87ce.png)

## Results

### Task 1 

#### RestNet 50 Model

Test accuracy= 98.99%, recall= 97-100%, precision= 98-100%

False negative= 0% piyuitary, 0.2% glioma, 0.9% menigioma

#### Vgg16 Model

Test accuracy= 97.12%, recall 92-99%, precision= 96-99%.

False negative= 1% pituitary, 0.3% glioma, 1.9% meningioma.

### Task 2

![image](https://user-images.githubusercontent.com/53411455/121937900-2bf0a600-cd19-11eb-8154-3eadbb625b33.png)

### Application of deep learning based semantic segmentation model

Random from the test data:

![image](https://user-images.githubusercontent.com/53411455/121938322-a7525780-cd19-11eb-9ed5-de5df7444cac.png)

## Conclusion

Both RESNET50 and vgg16 performed good for the multimodal classification of brain tumors. RESTNET50 model has 99-100% recall, among which 98-100% are true brain tumors.

The ReUNet, Dynamic UNet and Onet architecture gave the best dice score of 0.76,  0.799, 0.800 respectively. ResUnet is better for this type of data than the basic Unet and Deep ResUnet. It converts fatser and generate less noisy test accuracy. Onet and Dynamic UNet was performing the best with 0.8 dice score. More complex models do not always improve the performance of the network as Deep ResUnet or ResUNet.


## Business Value

Merits of automatic identification and segmention machine learning models are of extreme importance:

1) fast, accurate, automatic detection, and segmentation process.

2) Reliable.

3) Efficient treatment.

4) Precise traget of tumor cells for radiation treatment.

5) Reduce significantly the time of diagnosis.

6) Decrease the workload on healthworkers.

7) Decrease the number of healthworkers.

8) Prevent cancer spread and save lives

## Recommendations

1) Use deep learning CNN models for early detection and treatment of brain tumors.

2) Incorporate automatic identification of tumor models with MRI scan technology.

3) Incorporate automatic segemntation model into radiology treatment technology.

4) Incorporate segemntation maps for surgery operation or robotic surgery arms.

5) App made available for everyone to visualiza their MRI brain results.

## Future work

1) Develop a segmentation model for diffrent stages of glioma.

2) Improve the architecture of CNN models to achieve a dice score greater than 0.8.

3) Apply diffrent models architecture on brain tumor dataset with optimization:

e.g. Attention Unet, 3D ResUNet, Multi ResUNet, TransUNet, UNet 3+ , UNet++, PSPNet, DeepLab V3+.

## Refernces

Md Zahangir Alom et al. “Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation”. In: arXiv (Feb. 2018). eprint: 1802.06955. url: https://arxiv.org/abs/1802. 06955.

Brain tumor dataset. [Online; accessed 27. Sep. 2019]. Apr. 2017. url: https://figshare.com/articles/brain_tumor_dataset/1512427/5.

Kaiming He et al. “Identity Mappings in Deep Residual Networks”. In: arXiv (Mar. 2016). eprint: 1603.05027. url: https://arxiv.org/abs/ 1603.05027.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation”. In: arXiv (May 2015). eprint: 1505.04597. url: https://arxiv.org/abs/1505.04597.

Zhengxin Zhang, Qingjie Liu, and Yunhong Wang. “Road Extraction by Deep Residual U-Net”. In: arXiv (Nov. 2017). doi: 10.1109/LGRS.2018.

Wang, C.; Li, C.; Liu, J.; Luo, B.; Su, X.; Wang, Y.; Gao, Y. U2-ONet: A Two-Level Nested Octave U-Structure Network with a Multi-Scale Attention Mechanism for Moving Object Segmentation. Remote Sens. 2021, 13, 60. https://doi.org/10.3390/rs13010060

F. Isenee et al, Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenege. Computer Vision and Pattern Recognition. arXiv: 1802.10508v1.















