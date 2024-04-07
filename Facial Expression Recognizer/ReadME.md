# Data Source - 
Kaggle.com 
Huggingface.co

# About the Dataset - 
We gathered two emotion datasets for our project. The first one is from the FER competition, accessible via Kaggle, with seven emotion categories. The second one, from Hugging Face, has eight emotions with different folder labels and no predefined train-test split. We will reclassify the 'Contempt' category and ensure consistency in the folder names and data organization to enable effective preprocessing.

# Preprocessing - 
Consistency of input data is crucial for effectively training deep learning models in the realm of machine learning and computer vision. The High Fidelity (Hugging Face) dataset initially contains images with a resolution of 96x96 pixels, which differs from the Facial Expression Recognition (FER) dataset, comprising grayscale images with a resolution of 48x48 pixels. The first step in data preprocessing involves transforming the HF dataset images to match the FER dataset's resolution and color scheme. This transformation is critical as it ensures that the neural network receives uniform input data in both size and color depth.

# Pipeline - 
Data Collection -> Data Preprocessing -> Transfer Learning (Pre-trained architectures) -> Integrate with fine tuned YOLO v5

# Architecture Used - 
ResNet 
AlexNet
VGG-16
MobileNet

#Use Cases: 
1) Office Moral Monitoring - This implementation can be used to monitor the office environment, and certain threshold could be set, below which it could trigger an alert to the higher management to take required actions.
2) People suffering from a type of autism, where it is hard for people to understand social cues, sarcasm, irony and other forms of communications.

# Results 
We developed five models for our project, out of which four were trained using transfer learning through pre-trained architectures mentioned above and one was built from scratch using the classical CNN architecture. The results from the baseline model were very poor compared to the other models. The model that performed the best was the one trained using VGG-16, followed by ResNet and AlexNet. On the other hand, the MobileNet model struggled to achieve an accuracy better than 54%.

# Contributors - 
Diana Catalina Lopera
Juan Henao Barrios
Harsh Udaybhai Bhatt
Rahul -> Me:)
Robert Kaczur
Zarina Dossaeyva
 

