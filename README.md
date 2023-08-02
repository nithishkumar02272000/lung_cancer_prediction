# Lung Cancer Prediction using Chest Scan Images
This repository contains a Python notebook that demonstrates how to use deep learning models (InceptionV3 and ResNet50) for predicting lung cancer types from chest scan images.

## Dataset

The dataset used in this project is available on Kaggle and can be found here : https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

## Installation
To run the notebook, you need to install the required libraries. You can do this by running the following commands:
  - pip install opendatasets
  - pip install pandas
  - pip install matplotlib
  - pip install tensorflow

## Usage
You can also use the trained models to make predictions on your own chest scan images. To do this, you can use the chestScanPrediction function provided in the notebook. Simply pass the path of your image as an argument to the function, along with the desired model.
  Example usage
    path = "path_to_your_chest_scan_image.png"
    chestScanPrediction(path, model_incep)  # Use `model_incep` or `model_resnet` as the model argument
Note: Please make sure to adjust the path variable to the location of your image.

## Model Comparison
The project includes two models: InceptionV3 and ResNet50. Below is a comparison of their accuracy on the test set:
  algos = ['Resnet50','InceptionV3']
  accuracy = [accuracy_resnet, accuracy_incep]
  accuracy = np.floor([i * 100 for i in accuracy])
  plt.figure(figsize=(6, 5))
  plt.bar(algos, accuracy, color='blue', width=0.4)
  plt.xlabel("Algorithms Applied")
  plt.ylabel("Accuracy")
  plt.title("Model Comparison")
  plt.show()

## Confusion Matrix
The confusion matrix shows the performance of the model on different lung cancer types:
  cm = confusion_matrix(test_data.classes, y_pred)
  plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Nithish Kumar K S
