# DEEP-RELATION
finding relations of two individuals and similarity score
 Deep Relation – Finding Blood Relationships using Face Images. In this era of advanced technology, the integration of artificial intelligence (AI) and facial recognition has opened new avenues for various applications, including biometric authentication, surveillance, and personalization. One such application is the identification of blood relations through facial images, which holds significant potential in diverse fields such as forensic science, genealogy, and social networking. Our project, Deep Relation, aims to leverage deep learning techniques to accurately infer blood relations solely from facial images. We propose a novel approach that combines convolutional neural networks (CNNs) for feature extraction and relational reasoning mechanisms for inferring complex familial relationships. To achieve this, we employ a large-scale dataset of annotated facial images representing individuals across multiple generations and diverse ethnicities. We preprocess the data to mitigate variations in lighting, pose, and facial expressions, thus enhancing the model's generalization capabilities. Subsequently, we design a multi-task learning framework to jointly predict facial attributes and infer familial relations, enabling the model to leverage shared representations and improve overall performance.

OBJECTIVE The objective of the project "Deep Relation: Utilizing Facial Recognition for Blood Relation Identification" is to develop an advanced system capable of accurately identifying and inferring blood relations solely from facial images.

DATASETS: 1.KinFaceW-I and KinFaceW-II(used in Developing and Training a VGG16 model.) [There are four kin relations in two datasets: Father-Son (F-S), Father-Daughter (F-D), Mother-Son (M-S), and Mother-Daughter (M-D). In the KinFaceW-I dataset, there are 156, 134, 116, and 127 pairs of kinship images for these four relations. For the KinFaceW-II dataset, each relation contains 250 pairs of kinship images.] 2.Families in wild(Used in pre-trained model (Inception ResnetV1)) [FIW (Families In The Wild) FIW is a large and comprehensive database available for kinship recognition. FIW is made up of 11,932 natural family photos of 1,000 families-- nearly 10x more than the next-to-largest, Family-101 database.]

PROJECT WORKFLOW We had applied two approaches to accomplish the project as desired according to the problem statement: -  Developing and Training a VGG16 model.  Using a pre-trained model (Inception ResnetV1).


5.1-INCEPTION RESNET V1: Inception ResNet v1 is a convolutional neural network architecture developed to excel in image classification tasks. Combining the strengths of both the Inception and ResNet architectures, it features inception modules with residual connections, allowing for efficient information flow and deeper network training while mitigating the vanishing gradient problem. With its intricate design, Inception ResNet v1 achieves impressive performance on various benchmark datasets, providing state-of-the-art results in image recognition tasks by leveraging parallel and multi-scale feature extraction, enabling robust representation learning, and facilitating more accurate predictions with relatively lower computational costs compared to other contemporary architectures.

TRAINING A VGG16 MODEL:(code(Untitled1.ipynb) 1.1 Loading the VGG16 Model: Using an open-source neural network library which is written in python named KERAS has a keras Applications module that consists of various pre-trained models which are trained on the ImageNet Dataset. So, we use this module to load the VGG16 model that must be trained on our dataset (KinFaceW-1).

1.2 Importing all the required Libraries: 1.3 Loading the Data to the Model: 1.4- Training 1.4- Data Collection And Preparation The dataset used is “kinface”. Kinship is made up of 1068 Natural family photos of four relations. It is the largest and most comprehensive database available for kinface recognization. It consists of 524 image pairs split between four relationships. In this project preprocessing the data is not necessary much because the dataset is already too big and comprehensive even though minimal data preprocessing, such as resizing is completed. Screenshot 2024-04-25 184718

Drawbacks of using VGG16: -  It has only 16 layers that are not appropriate for calculating the similarity between the images accurately.  Inception ResNetV1 can achieve higher accuracy on tasks like image classification than VGG16 model.

USING A INCEPTION RESNETV1:(code(vggface.ipynb) Importing Libraries: The script imports necessary libraries including os, torch, PIL, numpy, facenet_pytorch, and matplotlib.pyplot.

Function Definitions:

get_all_image_paths(folder_path): Retrieves paths of all image files (.png, .jpg, .jpeg) within a specified folder.

display_result(input_image_path, folder_image_paths, similarity_scores, output_folder): Displays the input image and the most similar images from the folder along with their similarity scores. It also saves the result as an image file.

save_output_to_text(output_file, input_image_path, highest_similarity_folder, highest_similarity, folder_image_paths, similarity_scores): Saves the output details including input image path, most similar folder, similarity score, and details of folder images with their similarity scores to a text file.

get_facial_features(image_path): Extracts facial features from an image using MTCNN and InceptionResnetV1 models.

calculate_similarity(features1, features2): Calculates the cosine similarity between two sets of facial features.

get_highest_similarity_folder(input_image_path, root_folder_path): Determines the folder containing images most similar to the input image based on facial similarity. Main Execution:

Defines a list of input image paths, root folder path, output folder path, and output file path.

Iterates through each input image:

Calculates the most similar folder and its similarity score.

Displays the results.

Saves the output to a text file.

Execution:

The script executes the main part of the code if it's run as the main program (name == "main").

Input and Output Paths: Input image paths and root folder paths are specified. The output is saved in both image and text formats in the specified output folder.

Facial Similarity Calculation: The script uses pre-trained models to extract facial features from images and computes the similarity score between the input image and each image in the root folder. Overall, this script provides a comprehensive solution for comparing facial similarity between an input image and a set of images in a folder, aiding in tasks such as facial recognition, classification, or search.

![image](https://github.com/shreyaadepu13/DEEP-RELATION/assets/168128092/557c848a-e376-46f6-a66b-18186e88aaee)


![image](https://github.com/shreyaadepu13/DEEP-RELATION/assets/168128092/5d73af92-74fd-4bbd-893c-ccbdb6cc8b8d)
