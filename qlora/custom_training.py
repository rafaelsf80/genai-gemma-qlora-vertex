""" Custom training pipeline for Gemma 7B (PEFT), with script located at 'trainer.py'
    A custom container is used, located at TRAINING_IMAGE in Artifact Registry
"""

from datetime import datetime
from google.cloud import aiplatform

HF_TOKEN        = "YOUR_HUGGINGFACE_TOKEN"  # <--- CHANGE THIS !!

BUCKET          = 'gs://argolis-vertex-europewest4'
PROJECT_ID      = 'argolis-rafaelsanchez-ml-dev'
LOCATION        = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE_NAME = 'projects/989788194604/locations/europe-west4/tensorboards/8884581718011412480'
TRAINING_IMAGE  = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/gemma-qlora:latest'
TIMESTAMP       = datetime.now().strftime("%Y%m%d%H%M%S")

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)
                
job = aiplatform.CustomContainerTrainingJob(
    display_name="gemma_7b_qlora_gpu_" + TIMESTAMP,
    container_uri=TRAINING_IMAGE,
    #command=["python3", "trainer.py"],
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest"
)

env_vars = {
    "HF_TOKEN": HF_TOKEN
}

model = job.run(
    model_display_name="gemma_7b_qlora_gpu_" + TIMESTAMP,
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    environment_variables=env_vars,   
    tensorboard = TENSORBOARD_RESOURCE_NAME,
    machine_type="g2-standard-12",
    accelerator_type="NVIDIA_L4",
    accelerator_count = 1,
)
print(model)


# model_name = 'projects/989788194604/locations/europe-west4/models/7253491402577805312'
# model = aiplatform.Model(model_name)
# #Deploy endpoint
# endpoint = model.deploy(machine_type="g2-standard-12",
#     accelerator_type="NVIDIA_L4",
#     accelerator_count = 1)
# print(endpoint.resource_name)

# text= "Write me a poem about Machine Learning."
# response = endpoint.predict([[str(text)]])
# print(response)
# print("separo")
# print(response.predictions[0])

