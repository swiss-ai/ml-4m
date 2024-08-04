---
license: apache-2.0
tags:
- video LLM
datasets:
- OpenGVLab/VideoChat2-IT
---


# PLLaVA Model Card
## Model details
**Model type:** 
PLLaVA-7B is an open-source video-language chatbot trained by fine-tuning Image-LLM on video instruction-following data. It is an auto-regressive language model, based on the transformer architecture. Base LLM: llava-hf/llava-v1.6-vicuna-7b-hf

**Model date:**
PLLaVA-7B was trained in April 2024.

**Paper or resources for more information:**
- github repo: https://github.com/magic-research/PLLaVA
- project page: https://pllava.github.io/
- paper link: https://arxiv.org/abs/2404.16994

## License
llava-hf/llava-v1.6-vicuna-7b-hf license.

**Where to send questions or comments about the model:**
https://github.com/magic-research/PLLaVA/issues

## Intended use
**Primary intended uses:**
The primary use of PLLaVA is research on large multimodal models and chatbots.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

## Training dataset
Video-Instruct-Tuning data of OpenGVLab/VideoChat2-IT
## Evaluation dataset
A collection of 6 benchmarks, including 5 VQA benchmarks and 1 recent benchmarks specifically proposed for Video-LMMs.
