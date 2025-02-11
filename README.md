# XAI_HiResCAM
This is the repository of the paper "High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using structural MRI Images"
# data available at kaggle
D Swainson Sujana. (2024). Resampled [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9108251


# To download data 
!pip install kaggle

!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

# place kaggle.json file in the current directory

!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d dswainsonsujana/Resampled

!unzip -q Resampled.zip
