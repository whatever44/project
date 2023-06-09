# project
Breast Cancer Detection using Machine Learning
This project aims to develop a breast cancer detection model using two popular machine learning algorithms, Support Vector Machines (SVM) and Logistic Regression. The model is implemented from scratch, and k-fold cross-validation technique is used to evaluate its performance.

Breast cancer is a common form of cancer that affects millions of people worldwide. Early detection plays a crucial role in improving survival rates and treatment outcomes. Machine learning techniques provide a powerful tool to analyze medical data and assist in the early detection of breast cancer.

Dataset
The model utilizes a breast cancer dataset, which contains various features extracted from digitized images of fine needle aspirates (FNA) of breast masses. The dataset includes a set of input features and corresponding class labels indicating whether a given sample is malignant (cancerous) or benign (non-cancerous).

Feature Extraction and Preprocessing
Before training the model, feature extraction and preprocessing steps are performed on the dataset. This involves tasks such as normalizing the data, handling missing values, and splitting the dataset into training and testing sets.

Support Vector Machines (SVM)
Support Vector Machines is a powerful supervised learning algorithm widely used for classification tasks. In this project, SVM is used to classify breast cancer samples as malignant or benign based on the extracted features. The SVM algorithm aims to find an optimal hyperplane that separates the two classes with the maximum margin.

Logistic Regression
Logistic Regression is another commonly used algorithm for binary classification problems. It models the probability of an input belonging to a specific class. In this project, logistic regression is employed to predict the likelihood of breast cancer based on the extracted features.

k-fold Cross-Validation
To evaluate the performance of the developed model, k-fold cross-validation is utilized. Cross-validation helps to assess the model's generalization capabilities and reduce the risk of overfitting. The dataset is divided into k equal-sized folds, and the model is trained and tested k times, with each fold serving as the testing set once.

Implementation
The breast cancer detection model is implemented from scratch using Python programming language and popular machine learning libraries such as scikit-learn and NumPy. The steps involved in building the model include:

Data loading and preprocessing.
Feature extraction and normalization.
Splitting the dataset into training and testing sets.
Training the SVM and logistic regression models using the training set.
Evaluating the model's performance using k-fold cross-validation.
Making predictions on the testing set and calculating accuracy, precision, recall, and F1-score.
