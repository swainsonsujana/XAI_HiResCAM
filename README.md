# XAI_HiResCAM
This is the repository of the paper "High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using structural MRI Images"
# Data available at 
D Swainson Sujana. (2024). Resampled [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9108251

The aim of the study is to interpret the positive predictions of an Autism diagnosis model. This study employed the explainable AI methods CAM++, Guided_CAM, and HiResCAM (High-resolution Class Activation Map) to interpret the model's predictions. At first, a basic CNN model was built to diagnose autism from structural MRI images. The model's decision on positive (Autistic) predictions is explained through the XAI methods. To check the correctness and to ensure the trustworthiness of the generated XAI explanations, this study employed two techniques: 1. Occluding the original image and 2. Progressive masking. Metrics such as faithfulness score and integrated Area Under Curve(iAUC) were used to evaluate the trustworthiness of the generated explanations. Faithful explanations are expected to have a higher Faithfulness score and a lower AUC value. In this study, we found that HiResCAM achieved the highest faithfulness score and lowest iAUC value, which proved to be a reliable explanation for the autism diagnosis model.
# Model Building
The 3D-sMRI images were resized and normalized before they actually fed into the model. A modified LeNet-based model was built with three convolutional layers. The train split ratio was set as 73:27. The model has been trained for 100 epochs with the early stopping mode enabled(patience=15). The AUC value of the model on the validation data is 0.8434.

![AUC](https://github.com/user-attachments/assets/322491d1-a200-4a72-bb78-4eb555f227bc)

# Explaining model's prediction
The post-hoc explanation was performed by taking a sample autistic image from the dataset. The trained model was loaded and the original image prediction was found.

![image](https://github.com/user-attachments/assets/26156e52-1133-42d7-a14b-c8f845367121)  ![image](https://github.com/user-attachments/assets/b112e2fb-b27b-4134-9c92-afcdc36b633b)



 Pid:30245


# Gradient map
At first the gradient map for the input image was generated

![image](https://github.com/user-attachments/assets/a7dd0b19-2e3e-4852-8dd8-4784a06ab13a)



# CAM++
Then the CAM++ was generated for the same image

![image](https://github.com/user-attachments/assets/555e8633-db30-4e66-9309-9ece892dcacc)



# Guided CAM
Next guided_CAM was generated 


![image](https://github.com/user-attachments/assets/c62ea691-0e0c-46d9-967d-3b931480c61c)



# HiResCAM
The Hi-resolution class activation map(HiResCAM) was generated from gradient map and from guided_CAM by performing element wise multiplication.


![image](https://github.com/user-attachments/assets/3b544ec0-b732-4e82-8fe8-f51490ed2bc6)



# Finding Faithfulness score(Occlusion method)
The generated visual explanations were tested for its faithfulness by finding the faithfulness score. The higher faithfulness score indicates the explanations are reliable and trustworhty. This is accomplishd through the method called image occlusion.
# Occlusion based on CAM++

![image](https://github.com/user-attachments/assets/0d924a4b-d9c1-4958-860e-af41105a9201)


# Occlusion based on Guided_CAM

![image](https://github.com/user-attachments/assets/3a5e9765-0b1b-4473-9435-a945e8fd0974)

# Occlusion based on HiResCAM

![image](https://github.com/user-attachments/assets/e9aa6a93-96d5-426b-96cf-9b78a85eacf9)


# Progressive masking
To measure the iAUC value of the generated explanations, we performed progressive masking of the input image based on the peak values detected in the explanations.
For a good explanation the iAUC value must be small(less than 0.5)
# iAUC on CAM++

![image](https://github.com/user-attachments/assets/81e3504e-7a85-4ae1-9a3b-c196e6e66940)

# iAUC on guided_CAM

![image](https://github.com/user-attachments/assets/0504ebf3-2ee2-4684-b2e2-35531d8f5231)

# iAUC on HiResCAM

![image](https://github.com/user-attachments/assets/6ae75738-6532-4433-95fa-101eea6a540e)










