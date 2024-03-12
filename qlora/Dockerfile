# gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/gemma-qlora

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# https://github.com/NVIDIA/nvidia-docker/issues/1009#issuecomment-1181312052
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# install pytorch and cuda
ARG PYTORCH='2.0.1'
ARG VERSION='torch=='$PYTORCH'.*'
# Example: `cu102`, `cu113`, 'nightly/cu121', etc. ctransformers needs cuda 12+
ARG CUDA='nightly/cu121'
RUN python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN pip install -q -U bitsandbytes==0.42.0
RUN pip install -q -U peft==0.8.2
RUN pip install -q -U trl==0.7.10
RUN pip install -q -U accelerate==0.27.1
RUN pip install -q -U datasets==2.17.0
RUN pip install -q -U transformers==4.38.1
RUN pip install google-cloud-aiplatform google-cloud-storage

COPY trainer.py trainer.py

# Sets up the entry point to invoke the trainer
ENTRYPOINT ["python3", "trainer.py"]