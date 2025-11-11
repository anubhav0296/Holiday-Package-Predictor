# 🏖️ Holiday Package Predictor

An **end-to-end Machine Learning project** that predicts whether a customer will purchase a holiday package based on their demographic and behavioral data.  
The project is **fully deployed on AWS**, demonstrating the complete MLOps workflow — from data ingestion to model deployment.

---

## 🚀 Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Framework:** Flask (for web app)  
- **Deployment:** AWS Elastic Beanstalk / CodePipeline / S3  
- **Version Control:** Git & GitHub  


## ⚙️ Project Workflow

1. **Data Ingestion** – Collected and prepared customer data for training  
2. **Data Transformation** – Encoded categorical variables, handled nulls, and scaled numeric features  
3. **EDA (Exploratory Data Analysis)** – Identified customer trends and key features influencing purchase decisions  
4. **Model Training** – Built classification models (Logistic Regression, Random Forest, Gradient Boosting, etc.)  
5. **Model Evaluation** – Compared models using Accuracy, F1-score, ROC-AUC  
6. **Model Deployment** – Deployed the best model as a Flask app on AWS  

☁️ Deployment (AWS)

Deployed Flask app on AWS Elastic Beanstalk

Model artifacts stored in Amazon S3

Continuous deployment setup using AWS CodePipeline

📊 Model Performance

Achieved ~90–93% accuracy on test dataset

Fine-tuned hyperparameters using RandomizedSearchCV

Model identifies high-likelihood customers for package purchases
