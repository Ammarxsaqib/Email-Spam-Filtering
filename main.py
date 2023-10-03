import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np

# Load the data with explicit encoding
data = pd.read_csv('tested.csv', encoding='ISO-8859-1')

# Preprocess the text data (remove punctuation, convert to lowercase, tokenize, etc.)
# You can use your own text preprocessing code here.

# Encode the labels
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC()

# Create a pipeline with TF-IDF and SVM
from sklearn.pipeline import Pipeline
model = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('svm', svm_classifier)
])

# Define hyperparameters for grid search
param_grid = {
    'tfidf__max_features': [5000, 10000, 20000],  # Vary the number of features
    'svm__C': [1, 10, 100],  # Regularization parameter
    'svm__kernel': ['linear', 'rbf'],  # Kernel function
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['ham', 'spam'])
plt.yticks(tick_marks, ['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate and plot ROC curve
y_score = best_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print("Best Model Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
