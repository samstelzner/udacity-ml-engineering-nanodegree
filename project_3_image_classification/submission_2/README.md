# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Dataset
Dogbreed classification dataset.

### Access
Data uploaded to S3 for use in Sagemaker.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.
- ResNet 18. After initially exploring the use of DenseNet121, based on it performing well in  [this article](https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a) and being wholly different from ResNet, which I was exposed to in the course, I reverted to using ResNet18, the model I'm most familiar with, to hasten progress.
- Parameter 1: Learning rate. Range between 0.001 and 0.1.
- Parameter 2: Batch size. Range is one of the following: [32, 64, 128, 256, 512].

Inclusions:
- [Hyperparameter ranges](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/submission_images/Hyperparameter%20ranges.png)
- [Best hyperparameters](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/submission_images/Best%20hyperparameters.png)
- [Completed training jobs](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/submission_images/Completed%20training%20jobs.png)
- [Logs metrics during the training process](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/submission_images/Logs%20metrics%20during%20the%20training%20process.png)

## Debugging and Profiling
Model debugging and profiling in Sagemaker:
- Using train_model.py,
- Hooks defined in main function
- and called in train and test functions, where the hook modes are set to 'train' and 'eval' respectively.
- This creates tensors that represent the training state at each point in the training lifecycle ([ref.](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)).
- The parameters for these hooks are defined in our notebook using DebuggerHookConfig and then included in our estimator.
- They run when calling fit on our estimator.

### Results
Results summarised in [profiler-report](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/profiler-report.html)


## Model Deployment
Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
- The trained model is saved to S3 in a path such as: "s3://sam-dl-project/train_model/pytorch-training-2022-12-26-19-41-57-695/output/model.tar.gz".
- A predictor is deployed to an endpoint with the use of [PyTorchModel](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html) and an inference.py file as its entry point. The inference file loads the model, deserializes an input and returns a prediction.
- We can then call predict on our predictor with a serialised input (sample image) as argument.

[Deployed active endpoint in Sagemaker](https://github.com/samstelzner/udacity-ml-engineering-nanodegree/blob/main/project_3_image_classification/submission_2/submission_images/Deployed%20active%20endpoint%20in%20Sagemaker.png).