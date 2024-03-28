# Import necessary libraries
import streamlit as st
import boto3
import io
from PIL import Image
import json
import base64

# Initialize boto3 client for SageMaker
sagemaker = boto3.client('sagemaker-runtime', region_name='us-east-2')

# Streamlit app
def main():
    st.title("Group 6: Text to Image Generator - Stable Diffusion 2.1")

    # User input
    user_input = st.text_input("Enter a description (e.g., 'chest xray with pneumonia'):")

    if user_input:
        # Prepare the payload
        payload = {
            "prompt": user_input,
            "width": 400,
            "height": 400,
            "num_images_per_prompt": 1,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
        }

        # Invoke SageMaker endpoint with the Accept header set to 'application/json;jpeg'
        response = sagemaker.invoke_endpoint(
            EndpointName='jumpstart-ftc-stable-diffusion-v2-1-base',
            Body=json.dumps(payload),
            ContentType='application/json',
            Accept='application/json;jpeg'
        )

        # Parse the response
        response_dict = json.loads(response['Body'].read())
        
        # Extract the first image data from the 'generated_images' list
        image_base64 = response_dict['generated_images'][0]
        
        # Convert base64 encoded image to bytes
        image_bytes = base64.b64decode(image_base64)

        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Display image
        st.image(image, caption='Generated Image', use_column_width=True)

if __name__ == '__main__':
    main()