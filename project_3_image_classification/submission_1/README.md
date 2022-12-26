# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Dataset
Dogbreed classification dataset.

### Access
Data uploaded to S3 for use in Sagemaker.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.
- ResNet 18.
- Parameter 1: Learning rate. Range between 0.001 and 0.1.
- Parameter 2: Batch size. Range is one of the following: [32, 64, 128, 256, 512].

Inclusions:
- !(Hyperparameter ranges)[https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_1/submission_images/hyperparameter%20ranges.png]
- !(Best hyperparameters)[https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_1/submission_images/best%20hyperparameters.png]

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
