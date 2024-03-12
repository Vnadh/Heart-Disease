# Heart Disease Prediction using K-Nearest Neighbors (KNN) Algorithm

Name RollNo
VALLEPU VEERENDRA NADH 21761A0558
THOTA GOPISANKAR 21761A0557
THOTA ANJIBABU 21761A0556

## Abstract:
The core idea of the project is to create a predictive model for diagnosing heart disease in individuals
based on their medical attributes. This project is significant as it addresses a critical healthcare issue
and demonstrates the application of machine learning techniques in the medical field. The
accomplished goal of the project is to develop a KNN algorithm that can accurately predict the
presence or absence of heart disease in a patient. The key outcomes include the accuracy of the
model, the confusion matrix, and the classification report, all of which will be discussed in the
README.
## Introduction:
Heart disease is a leading cause of mortality worldwide, and early diagnosis is crucial for effective
treatment. This project uses the K-Nearest Neighbors (KNN) algorithm to predict the presence of
heart disease based on a set of medical attributes. By analyzing a dataset of patients with known
heart conditions, the project aims to create a reliable tool for diagnosing heart disease.
The relevance of this project lies in its potential to assist healthcare professionals in making accurate
diagnoses, particularly in cases where early intervention can be life-saving. Furthermore, it
showcases the power of machine learning in solving real-world problems, emphasizing its broader
applications beyond traditional data analysis.
## Methodology:
The project begins by importing the necessary Python libraries, including pandas, seaborn,
matplotlib, and scikit-learn. It then loads a dataset containing medical information of patients, which
will serve as the basis for training and testing the KNN model.
The data is visualized to understand the distribution of age among patients and to explore the
relationship between age and the presence of heart disease. The data is then split into training and
testing sets. A KNN classifier is trained using the training data, and predictions are made on the
testing data.
## Results:
The project's results are as follows:
• Accuracy: The accuracy of the KNN model is a measure of its correctness in predicting the
presence or absence of heart disease.
Accuracy: 0.6885245901639344

• Confusion Matrix: The confusion matrix provides information about the true positive, true
negative, false positive, and false negative predictions. It is a valuable tool for evaluating the
model's performance.
• Confusion Matrix:
• [[18 11]
• [ 8 24]]
• Classification Report: The classification report includes metrics such as precision, recall, F1-
score, and support for both classes (presence and absence of heart disease). It provides a
more comprehensive assessment of the model's performance.
• Classification Report:
• precision recall f1-score support
•
• 0 0.69 0.62 0.65 29
• 1 0.69 0.75 0.72 32
•
• accuracy 0.69 61
• macro avg 0.69 0.69 0.69 61
• weighted avg 0.69 0.69 0.69 61
## Conclusion:
In this project, a KNN algorithm was employed to predict heart disease in individuals. The model's
accuracy and performance were evaluated using a confusion matrix and a classification report. The
results of this project demonstrate the potential of machine learning techniques in aiding medical
professionals in diagnosing heart disease accurately.
This project's implications extend beyond the specific use case, illustrating how data science and
machine learning can have a significant impact on healthcare and contribute to early disease
detection.
## References:
• Scikit-Learn Documentation
• Seaborn Documentation
• Matplotlib Documentation
• Dataset Source:
https://colab.research.google.com/drive/1VYT09mOKXWu64mrBx6CjhMogiASBehXt
