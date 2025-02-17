# XAI_HiResCAM
This is the repository of the paper "High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using structural MRI Images"
# data available at kaggle
D Swainson Sujana. (2024). Resampled [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9108251

The aim of the study is to evaluate the explainable AI methods CAM++, Guided_CAM and HiResCAM(High resolution Class Activation Map). This study built a basic CNN model to diagnose autism from structural MRI images. The model's decision on positive(Autistic) predictions are explained through the XAI methods. To check the correctness and to ensure the trustworthiness of the generated XAI explanations, this study employed two techniques: 1. occluding the original image 2. Progressive masking. Metrics such as faithfulness score and integrated Area Under Curve(iAUC) were used to evaluate the trustworthiness of the generated explanations. The higher Faithfulness score and the lower iAUC value reveals the hiprecision explanation for the model's prediction. In this study we found HiResCAM achieves high faithfulness score and lower iAUC value and proved to be the reliable explantion for the autism diagnosis model. 

#Model Building
A modiifed LeNet based model was built with 



