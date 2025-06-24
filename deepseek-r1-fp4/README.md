---
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-R1-0528
license: mit
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- DeepSeekR1
- quantized
- FP4
---
# Model Overview

## Description:
The NVIDIA DeepSeek-R1-0528-FP4 model is the quantized version of the DeepSeek AI's DeepSeek R1 0528 model, which is an auto-regressive language model that uses an optimized transformer architecture. For more information, please check [here](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528). The NVIDIA DeepSeek R1 FP4 model is quantized with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

This model is ready for commercial/non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(DeepSeek R1) Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528).

### License/Terms of Use:
[MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)


## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** DeepSeek R1 <br>

## Input:
**Input Type(s):** Text <br>
**Input Format(s):** String <br>
**Input Parameters:** 1D (One Dimensional): Sequences <br>
**Other Properties Related to Input:** DeepSeek recommends adhering to the following configurations when utilizing the DeepSeek-R1 series models, including benchmarking, to achieve the expected performance: \

- Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
- Avoid adding a system prompt; all instructions should be contained within the user prompt.
- For mathematical problems, it is advisable to include a directive in your prompt such as: "Please reason step by step, and put your final answer within \boxed{}."
- When evaluating model performance, it is recommended to conduct multiple tests and average the results. <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** 1D (One Dimensional): Sequences <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* TensorRT-LLM <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

## Model Version(s):
** The model is quantized with nvidia-modelopt **v0.31.0**  <br>

## Training Dataset: <br>
** Data Collection Method by dataset: Hybrid: Human, Automated <br>
** Labeling Method by dataset: Hybrid: Human, Automated <br>

## Testing Dataset: <br>
** Data Collection Method by dataset: Hybrid: Human, Automated <br>
** Labeling Method by dataset: Hybrid: Human, Automated <br>

## Evaluation Dataset: <br>
** Data Collection Method by dataset: Hybrid: Human, Automated <br>
** Labeling Method by dataset: Hybrid: Human, Automated <br>

## Calibration Datasets:
* Calibration Dataset: [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) <br>
** Data collection method: Automated. <br>
** Labeling method: Undisclosed. <br>

## Inference:
**Engine:** TensorRT-LLM <br>
**Test Hardware:** B200 <br>

## Post Training Quantization
This model was obtained by quantizing the weights and activations of DeepSeek R1 to FP4 data type, ready for inference with TensorRT-LLM. Only the weights and activations of the linear operators within transformer blocks are quantized. This optimization reduces the number of bits per parameter from 8 to 4, reducing the disk size and GPU memory requirements by approximately 1.6x.

## Usage

### Deploy with TensorRT-LLM

To deploy the quantized FP4 checkpoint with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) LLM API, follow the sample codes below (you need 8xB200 GPU and TensorRT LLM built from source with the latest main branch):

#### LLM API sample usage:
```
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM

def main():

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model="nvidia/DeepSeek-R1-0528-FP4", tensor_parallel_size=8, enable_attention_dp=True)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()

```


#### Minimum Latency Server Deployment

If you want to deploy your endpoint to minimize response latency for a single-concurrency or low-concurrency use case, follow the instructions below.

**Step 1: Create configuration file (`args.yaml`)**

```yaml
moe_backend: TRTLLM
use_cuda_graph: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
  use_relaxed_acceptance_for_thinking: true
  relaxed_topk: 10
  relaxed_delta: 0.6
```

**Step 2: Start the TensorRT-LLM server**

```bash
trtllm-serve nvidia/DeepSeek-R1-0528-FP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 4 \
  --tp_size 8 \
  --ep_size 2 \
  --max_num_tokens 32768 \
  --trust_remote_code \
  --extra_llm_api_options args.yaml \
  --kv_cache_free_gpu_memory_fraction 0.75
```

**Step 3: Send an example query**

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/DeepSeek-R1-0528-FP4",
    "messages": [{"role": "user", "content": "Why is NVIDIA a great company?"}],
    "max_tokens": 1024
  }'
```

### Evaluation
The accuracy benchmark results are presented in the table below:
<table>
  <tr>
   <td><strong>Precision</strong>
   </td>
   <td><strong>MMLU Pro</strong>
   </td>
   <td><strong>GPQA Diamond</strong>
   </td>
   <td><strong>LiveCodeBench</strong>
   </td>
   <td><strong>SCICODE</strong>
   </td>
   <td><strong>MATH-500</strong>
   </td>
   <td><strong>AIME 2024</strong>
   </td>
  </tr>
  <tr>
   <td>FP8 (AA Ref)
   </td>
   <td>85
   </td>
   <td>81
   </td>
   <td>77
   </td>
   <td>40
   </td>
   <td>98
   </td>
   <td>89
   </td>
  </tr>
  <tr>
   <td>FP4
   </td>
   <td>84.2
   </td>
   <td>80.0
   </td>
   <td>76.3
   </td>
   <td>40.1
   </td>
   <td>98.1
   </td>
   <td>91.3
   </td>
  </tr>
  <tr>
</table>

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## 运行推理

您可以使用以下代码片段进行推理。请注意，这仅是一个示例，您需要根据您的实际模型和数据进行调整。

```python
# import torch
# from deepseek_fp4_plugin.models.deepseek_v3 import DeepseekV3ForCausalLM
# from tensorrt_llm import SamplingParams
# from tensorrt_llm._torch import LLM

# # 示例：加载模型和运行推理
# model = DeepseekV3ForCausalLM.from_pretrained("your_model_path")
# llm = LLM(model, max_batch_size=1, max_new_tokens=100)

# inputs = ["Hello, world!"]
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_new_tokens=10)

# outputs = llm.generate(inputs, sampling_params)
# for output in outputs:
#     print(output.text)

print("此 README 文件仅作为 deepseek-r1-fp4 模型的示例说明。实际使用请参照主插件文档。")
```

