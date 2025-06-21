# Accelerating LLaMA Inference: 4-bit Quantization + TorchCompile Using PrunaAI
[In progress]

This project has applied inference optimizations on Meta’s LLaMA-3.2-1B-Instruct using Pruna AI — an optimization framework to make model faster, smaller, cheaper, greener.

## 𝗩𝗮𝗻𝗶𝗹𝗹𝗮 𝗠𝗼𝗱𝗲𝗹 𝗦𝗲𝘁𝘂𝗽
- Compute dtype: bfloat16
- Full Model Precision - FP32

## 𝗣𝗿𝘂𝗻𝗮𝗔𝗜 𝗢𝗽𝘁𝗶𝗺𝗶𝘇𝗮𝘁𝗶𝗼𝗻 𝗦𝗲𝘁𝘂𝗽
- Compute dtype: bfloat16
- 4-bit HQQ quantization (Hessian-aware)
- torch.compile with max-autotune + full graph mode

## 📊 𝗥𝗲𝘀𝘂𝗹𝘁𝘀
- Vanilla Inference Time: 0.754 sec
- Optimized Inference Time: 0.228 sec

→ That’s a 3.3× speedup on real hardware 🚀

![HF_vs_PrunaAI](https://github.com/user-attachments/assets/7ac3487d-1eb6-4de8-abb8-6f9be0856db0)


## Setup:
NVIDIA A40 (Runpod)
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

## References
- [Pruna AI Tutorials](https://github.com/PrunaAI/pruna/tree/main/docs/tutorials)
