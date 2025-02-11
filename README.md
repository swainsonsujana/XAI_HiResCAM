# XAI_HiResCAM
This is the repository of the paper "High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using structural MRI Images"
# data available in kaggle (doi:10.34740/kaggle/dsv/9108251)


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
