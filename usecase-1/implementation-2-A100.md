- We use 1XA100 GPUs @GCP with the Mistral 7B model in this experiment.

# Code Implementation

I will list only the differences here:

- Handling **multi-processers** had been **removed** as we are using a single GPU.
- **Quantization** had been **removed** as we are using the whole memory. 
