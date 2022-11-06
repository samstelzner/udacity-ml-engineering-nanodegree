# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Samuel Stelzner

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The Notebook prompted me to set any negative predicted values to zero. I only needed to do this with my 3rd submission (hyperparameter optimisation).

### What was the top ranked model that performed?
WeightedEnsemble_L3 for the initial and new feature submissions. WeightedEnsemble_L2 for the hpo submission.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
We plotted the different features as histograms showing, for example, a fairly normal distribution for temperature. I added hour as an additional feature, a part of the initially included datetime field.

### How much better did your model preform after adding additional features and why do you think that is?
The Kaggle score improved by a factor of 3. I would suspect that hour of the day matters a lot for bike sharing. For example, I would expect spikes around rush hour. So, including hour as a feature helped with performance.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
It was marginally better.

### If you were given more time with this dataset, where do you think you would spend more time?
Adding more features, from the initial datetime field, such as month of year. Also trying to better understand hyper paramater optimisation options.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
![hyperparameters_table.png](https://d-lqanqkvbc3iy.studio.us-east-1.sagemaker.aws/jupyter/default/files/nd009t-c1-intro-to-ml-project-starter/hyperparameters_table.png?_xsrf=2%7C72a5e37c%7Cb7905930bf2c5d1091ece486dbd004bf%7C1666443342)

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](https://d-lqanqkvbc3iy.studio.us-east-1.sagemaker.aws/jupyter/default/files/nd009t-c1-intro-to-ml-project-starter/model_train_score.png?_xsrf=2%7C72a5e37c%7Cb7905930bf2c5d1091ece486dbd004bf%7C1666443342)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](https://d-lqanqkvbc3iy.studio.us-east-1.sagemaker.aws/jupyter/default/files/nd009t-c1-intro-to-ml-project-starter/model_test_score.png?_xsrf=2%7C72a5e37c%7Cb7905930bf2c5d1091ece486dbd004bf%7C1666443342)

## Summary
We predicted bike share demand for days later in the month based on data from days earlier in the month using Autogluon's TabularPredictor. The data comprised of bike sharing and weather related fields. We improved our model by creating an hour feature from the datetime field and experimenting with some hyperparameter optimisation. Ultimately, the WeightedEnsemble type model proved the most effective.