![image](https://github.com/user-attachments/assets/41d595e5-3f42-4f6a-adf1-1a383f8a9ebb)# XAI_HiResCAM
This is the repository of the paper "High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using structural MRI Images"
# data available at kaggle
D Swainson Sujana. (2024). Resampled [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9108251

The aim of the study is to evaluate the explainable AI methods CAM++, Guided_CAM and HiResCAM(High resolution Class Activation Map). This study built a basic CNN model to diagnose autism from structural MRI images. The model's decision on positive(Autistic) predictions are explained through the XAI methods. To check the correctness and to ensure the trustworthiness of the generated XAI explanations, this study employed two techniques: 1. occluding the original image 2. Progressive masking. Metrics such as faithfulness score and integrated Area Under Curve(iAUC) were used to evaluate the trustworthiness of the generated explanations. The higher Faithfulness score and the lower iAUC value reveals the hiprecision explanation for the model's prediction. In this study we found HiResCAM achieves high faithfulness score and lower iAUC value and proved to be the reliable explantion for the autism diagnosis model. 

# Model Building
The 3D-sMRI images were resized and normalized before it actually fed into the model. A modified LeNet based model was built with 3 convoltuional layers. The train split ratio was set as 73:27. The model has been trained for 100 epochs with the early stopping mode enabled(patience=15). The AUC value of the model on the validation data is 0.8456. 

![AUC](https://github.com/user-attachments/assets/322491d1-a200-4a72-bb78-4eb555f227bc)

# Explaining model's prediction
The post-hoc explanation was performed by taking a sample autistic image from the dataset. The trained model was loaded and the original image prediction was found.

![image](https://github.com/user-attachments/assets/5c3ef390-a0aa-44a5-818c-c3f80ec0eea1) Pid:30245

# Gradient map
At first the gradient map for the input image was generated
![image](https://github.com/user-attachments/assets/da3d8b07-0dc0-4417-950c-b3b8ccf1ad8a)

# CAM++
Then the CAM++ was generated for the same image

![image](https://github.com/user-attachments/assets/270a5ccc-2c94-4b8b-b111-785e7a5c1962)

# guided CAM
Next visual explanation guided CAM was generated 

![image](https://github.com/user-attachments/assets/788264b4-a439-47cb-ac6b-545efd3e2a09)

# HiResCAM
The hiresolution class activation map(HiResCAM) was generated from gradient map and from guided CAM by performing element wise multiplication.

![image](https://github.com/user-attachments/assets/7c2d8a60-7f4c-42cb-9800-fe083cb6fc4e)





