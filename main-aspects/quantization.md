## What is Quantization?

Quantization is a compression technique that involes mapping high precision values to a lower precision one. For an LLM, that means modifying the precision of their weights and activations making it less memory intensive. This surely does have impact on the capabilites of the model including the accuracy. It is often a trade-off based on the use case to go with model that is quantized. It is found that in some cases its possible to achieve comparable results with significantly lower precision. Quantization improves performance by reducing memory bandwidth requirement and increasing cache utilization.

Instead of using high-precision data types, such as 32-bit floating-point numbers, quantization represents values using lower-precision data types, such as 8-bit integers. This process significantly reduces memory usage and can speed up model execution while maintaining acceptable accuracy.

With an LLM model, quantization process at different precision levels enables a model to be run on wider range of devices.

## How does quantization work?

LLMs are generally trained with full(float32) or half precision(float16 floating point numbers. One float16 has 16 bits which is 2 bytes. So it requires two gigabytes for one billion parameter model trained on FP16.

The process of quantization thus works on finding a way to represent the range (which is [min, max] for the datatype) of FP32 weight values to a lower precision values like FP16 or even INT4 (Integer 4 bit) datatypes. The typical case is one from FP32 to INT8.

The overall impact on the quality of LLM depends on the technique used.

## Hugging Face and Bitsandbytes Uses

Hugging Face’s Transformers library is a go-to choice for working with pre-trained language models. To make the process of model quantization more accessible, Hugging Face has seamlessly integrated with the Bitsandbytes library. This integration simplifies the quantization process and empowers users to achieve efficient models with just a few lines of code.

Install latest accelerate from source:

pip install git+https://github.com/huggingface/accelerate.git

Install latest transformers from source and bitsandbytes:

pip install git+https://github.com/huggingface/transformers.git

pip install bitsandbytes

![](https://miro.medium.com/v2/resize:fit:700/1*O4RAzlQkbrcCPiPPD9JIYw.jpeg)

Hugging Face and Bitsandbytes Integration Uses

## Loading a Model in 4-bit Quantization

One of the key features of this integration is the ability to load models in 4-bit quantization. This can be done by setting the  `load_in_4bit=True`  argument when calling the  `.from_pretrained`  method. By doing so, you can reduce memory usage by approximately fourfold.

from transformers import AutoModelForCausalLM, AutoTokenizer  
  
model_id = "bigscience/bloom-1b7"  
  
tokenizer = AutoTokenizer.from_pretrained(model_id)  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

## Loading a Model in 8-bit Quantization

For further memory optimization, you can load a model in 8-bit quantization. This can be achieved by using the  `load_in_8bit=True`  argument when calling  `.from_pretrained`. This reduces the memory footprint by approximately half.

from transformers import AutoModelForCausalLM, AutoTokenizer  
  
model_id = "bigscience/bloom-1b7"  
  
tokenizer = AutoTokenizer.from_pretrained(model_id)  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

You can even check the memory footprint of your model using the  `get_memory_footprint`  method:

print(model.get_memory_footprint())

# Other Use cases:

The Hugging Face and Bitsandbytes integration goes beyond basic quantization techniques. Here are some use cases you can explore:

## Changing the Compute Data Type

You can modify the data type used during computation by setting the  `bnb_4bit_compute_dtype`  to a different value, such as  `torch.bfloat16`. This can result in speed improvements in specific scenarios. Here's an example:

from transformers import BitsAndBytesConfig  
  
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

## Using NF4 Data Type

The NF4 data type is designed for weights initialized using a normal distribution. You can use it by specifying  `bnb_4bit_quant_type="nf4"`:

from transformers import BitsAndBytesConfig  
  
nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")  
  
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

## Nested Quantization for Memory Efficiency

The integration also recommends using the nested quantization technique for even greater memory efficiency without sacrificing performance. This technique has proven beneficial, especially when fine-tuning large models:

from transformers import BitsAndBytesConfig  
  
double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)  
  
model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)

## Loading a Quantized Model from the Hub

A quantized model can be loaded with ease using the  `from_pretrained`  method. Make sure the saved weights are quantized by checking the  `quantization_config`  attribute in the model configuration:

model = AutoModelForCausalLM.from_pretrained("model_name", device_map="auto")

In this case, you don’t need to specify the  `load_in_8bit=True`  argument, but you must have both Bitsandbytes and Accelerate library installed.

# Exploring Advanced techniques and configuration

There are additional techniques and configurations to consider:

## Offloading Between CPU and GPU

One advanced use case involves loading a model and distributing weights between the CPU and GPU. This can be achieved by setting  `llm_int8_enable_fp32_cpu_offload=True`. This feature is beneficial for users who need to fit large models and distribute them between the GPU and CPU.

## Adjusting Outlier Threshold

Experiment with the  `llm_int8_threshold`  argument to change the threshold for outliers. This parameter impacts inference speed and can be fine-tuned to suit your specific use case.

## Skipping the Conversion of Some Modules

In certain situations, you may want to skip the conversion of specific modules to 8-bit. You can do this using the  `llm_int8_skip_modules`  argument.

## Fine-Tuning a Model Loaded in 8-bit

With the support of adapters in the Hugging Face ecosystem, can fine-tune models loaded in 8-bit quantization, enabling the fine-tuning of large models with ease.

ref: https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996

https://medium.com/@techresearchspace/what-is-quantization-in-llm-01ba61968a51#:~:text=Quantization%20is%20a%20compression%20technique,the%20model%20including%20the%20accuracy.
