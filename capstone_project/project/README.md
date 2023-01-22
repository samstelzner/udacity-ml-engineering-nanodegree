# ML with AWS nanodegree capstone project

This is the submission folder for the capstone project of the Udacity ML with AWS nanodegree by Sam Stelzner.

The project aims to predict which passengers of the Titanic survived the tragic sinking in 1912, a renowned Kaggle competition.

Capstone Project Report.pdf is the detailed project report.

sam-capstone-project.ipynb, developed in AWS Sagemaker Studio, begins with exploring the provided passenger data.

Data Wrangler is used to process the data, transforming some values to create better features - see eda-data-wrangler.flow.

The outputs of the Data Wrangler flow is a processed train and test dataset. These are exported to S3 using the generated notebooks: eda.ipynb and eda1.ipynb.

A model is then developed and predictions made in sam-capstone-project.ipynb.

These predictions are submitted to the Kaggle competition to score the model accuracy.