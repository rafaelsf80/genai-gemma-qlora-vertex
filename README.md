#  Fine-tuning Gemma 7B on Vertex AI Training with QLoRA

This repo shows how to **fine-tune a Gemma 7B model** (8.54B parameters), using a `g2-standard-12` machine type with 1xL4 NVidia GPU in **Vertex AI Training**. 

The model is 4-bit quantized using [NF4](https://arxiv.org/abs/2305.14314) (QLoRA).

> PENDING: Model to be deployed on Vertex AI Prediction.


## The model: Gemma

[Gemma](https://www.kaggle.com/m/3301) is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. It's released with a [free license for commercial and educational use](https://ai.google.dev/gemma/terms).ddd

Gemma model family are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

Inputs and outputs:

* Input: Text string, such as a question, a prompt, or a document to be summarized.
* Output: Generated English-language text in response to the input, such as an answer to a question, or a summary of a document.

Available sizes:

* Gemma 2B: 18-layer model
* Gemma 2B-IT: 18-layer model with instruction tuning
* Gemma 7B: 28-layer model
* Gemma 7B-IT: 28-layer model with instruction tuning

Main features:

* English-only, text-only (not multimodal)
* Decoder-only transformer.
* 8k context length.
* 2T and 6T tokens (Gemma 2B and 7B respectively).

Model card [here](https://www.kaggle.com/models/google/gemma) and paper [here](http://goo.gle/GemmaReport).

Gemma 7B-IT model to be downloaded from [Hugging Face](https://huggingface.co/google/gemma-7b-it). 


## QLoRA

The dataset: Abirate/english_quotes

The dataset [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) is a dataset of all the quotes retrieved from goodreads quotes. This dataset can be used for multi-label text classification and text generation. The content of each quote is in English and concerns the domain of datasets for NLP and beyond.

Gemma [prompt format](https://ai.google.dev/gemma/docs/formatting) for instruction-tuned models (Gemma 2B-IT and Gemma 7B-IT):
```yaml
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
Gemma<end_of_turn>
<start_of_turn>model
Gemma who?<end_of_turn>
``` 

Commands for QLoRA tuning in Vertex AI Training:
```sh
gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/gemma-qlora
python3 custom_training.py 
```


## Inference

PENDING: Inference ot the finetuned model in Vertex AI Prediction


## References

* Codelab: [Showcasing Agile Safety Classifiers with Gemma](https://codelabs.developers.google.com/codelabs/responsible-ai/agile-classifiers)
* Codelab: [Using LIT to Analyze Gemma Models in Keras](https://codelabs.developers.google.com/codelabs/responsible-ai/lit-gemma)