import json
import base64

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-01-04-20-58-29-480'

# https://knowledge.udacity.com/questions/767689
import boto3
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])
    
    # Make a prediction:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image)
    
    # We return the data back to the Step Function    
    event['body']['inferences'] = json.loads(response['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event['body']
    }