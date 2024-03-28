# cvpipeline.py

import boto3
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join

#Enter your role here 'arn:aws:iam::....'
role = 'arn:aws:iam::365008225581:role/service-role/AmazonSageMaker-ExecutionRole-20230807T124543'
sagemaker_session = sagemaker.Session()

# Constants
input_folder = 's3://cv-project-bucket-1/archive.zip'
output_folder = 's3://cv-project-bucket-1/processed_data'


# Step 1: Data Processing using SageMaker Processing
# Define the SKLearn processor
sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                     role=role,
                                     instance_type='ml.m5.24xlarge',
                                     instance_count=1,
                                     sagemaker_session=sagemaker_session)

# Define inputs and outputs
input_data = ProcessingInput(source=input_folder, destination='/opt/ml/processing/input1')
output_data = ProcessingOutput(source='/opt/ml/processing/output1', destination=output_folder)


# Define the data processing step
processing_step = ProcessingStep(
    name="DataProcessing",
    processor=sklearn_processor,
    code='data_processing.py',
    inputs=[input_data],
    outputs=[output_data]
)


# Step 2: Model Training using SageMaker Training
estimator = TensorFlow(entry_point='train.py',
                       source_dir='s3://cv-project-bucket-1/code/',
                       role=role,
                       instance_count=1,
                       instance_type='ml.m5.24xlarge',
                       framework_version='2.3.0',
                       py_version='py37',
                       sagemaker_session=sagemaker_session)


train_data_uri = Join(on='/',
                   values=[processing_step.properties.ProcessingOutputConfig.Outputs['output-1'].S3Output.S3Uri, 
                           'chest_xray', 'train'])

validation_data_uri = Join(on='/',
                        values=[processing_step.properties.ProcessingOutputConfig.Outputs['output-1'].S3Output.S3Uri, 
                                'chest_xray', 'val'])



training_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        'train': sagemaker.TrainingInput(s3_data=train_data_uri, content_type='image/jpeg'),
        'validation': sagemaker.TrainingInput(s3_data=validation_data_uri, content_type='image/jpeg')
    }
)

# Create the pipeline
pipeline = Pipeline(
    name="CVPipeline",
    steps=[processing_step, training_step],
    sagemaker_session=sagemaker_session
)

# Submit the pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()

execution.wait()


# deploy the model as an endpoint immediately after training
# Step 3: Model Deployment using SageMaker Endpoint
# Provided S3 path to the trained model artifacts
model_artifacts = 's3://cv-project-bucket-1/model/best_model.tar.gz'

# Create a SageMaker Model from the model artifacts
model = TensorFlowModel(model_data=model_artifacts,
                        role=role,
                        framework_version='2.3.0',
                        py_version='py37',
                        sagemaker_session=sagemaker_session)

# Deploy the model
endpoint_name = "cv-endpoint"
predictor = model.deploy(endpoint_name=endpoint_name, initial_instance_count=1, instance_type='ml.m5.24xlarge')
