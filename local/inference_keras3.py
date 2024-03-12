""" Basic inference of Gemma 2B using Keras 3
    Inference with CPU takes several minutes
    IMPORTANT: You must accept Gemma license conditions on Kaggle page. Otherwise, you will get a 403 error when doing `.from_preset()` 
"""

#!pip install -q -U keras-nlp
#!pip install -q -U keras

import keras
import keras_nlp

import os

os.environ["KAGGLE_USERNAME"] = "YOUR_KAGGLE_USERNAME_HERE"
os.environ["KAGGLE_KEY"]      = "YOUR_KAGGLE_KEY_HERE"
os.environ["KERAS_BACKEND"]   = "tensorflow"  # Or "jax" or "torch".

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

gemma_lm.summary()

print(gemma_lm.generate("What is the meaning of life?", max_length=64))

# Second inference should be very fast, thanks to XLA and tensorflow/jax backends
print(gemma_lm.generate("How does the brain work?", max_length=64))

# Batch inference
print(gemma_lm.generate(
    ["What is the meaning of life?",
     "How does the brain work?"],
    max_length=64))

# Optionally try a different sampler
#gemma_lm.compile(sampler="top_k")
#gemma_lm.generate("What is the meaning of life?", max_length=64) #, top_k=3)
