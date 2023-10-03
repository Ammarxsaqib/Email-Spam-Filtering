# Email-Spam-Filtering-TechnoHacks
In this , we address the task of classifying text messages as either "ham" (not spam) or "spam" using machine learning techniques. This problem is essential for spam email filtering, fraud detection, and more.

Overview
In this project, we perform the following steps:

Data Loading: We start by loading the text message dataset from a CSV file (tested.csv) using the pandas library. The dataset contains text messages and their corresponding labels ("ham" or "spam").

Text Preprocessing: Text preprocessing is crucial for text classification tasks. Although not included in the code, you can apply your own text preprocessing techniques, such as removing punctuation, converting to lowercase, tokenization, and stemming/lemmatization, to prepare the text data for modeling.

Label Encoding: We encode the text labels ("ham" and "spam") as numeric values (0 and 1) to make them compatible with machine learning algorithms.

Data Splitting: We split the data into training and testing sets using the train_test_split function from scikit-learn. This separation allows us to train the model on one portion of the data and evaluate its performance on another.

TF-IDF Vectorization: We create a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer using TfidfVectorizer from scikit-learn. TF-IDF is a numerical statistic that reflects the importance of words in a document relative to a collection of documents. It is a commonly used technique for text-based features in text classification tasks.

SVM Classifier: We create a Support Vector Machine (SVM) classifier using SVC from scikit-learn. SVM is a powerful algorithm for binary classification tasks.

Pipeline Creation: We create a machine learning pipeline using Pipeline from scikit-learn. The pipeline consists of the TF-IDF vectorizer and the SVM classifier, allowing us to streamline the preprocessing and modeling steps.

Hyperparameter Tuning: We define a hyperparameter grid for grid search using GridSearchCV from scikit-learn. Hyperparameters like the number of TF-IDF features, the regularization parameter (C), and the kernel function are varied to find the best combination.

Model Training: We perform grid search with cross-validation to find the best combination of hyperparameters. The best model is selected based on cross-validation performance.

Model Evaluation: We evaluate the best model's performance using metrics such as accuracy, the classification report, and the confusion matrix. Additionally, we visualize the confusion matrix and plot the Receiver Operating Characteristic (ROC) curve to assess the model's ability to discriminate between classes.

Usage

Clone this repository:
bash
Copy code
git clone https://github.com/your-username/text-classification-svm.git

Navigate to the project directory:
bash
Copy code
cd text-classification-svm
Ensure you have the tested.csv file in the same directory as the code.

Run the Jupyter Notebook or Python script to execute the project.

Requirements

Before running the code in this repository, ensure that you have the necessary Python libraries installed. You can install them using pip:
bash
Copy code
pip install pandas scikit-learn matplotlib seaborn
