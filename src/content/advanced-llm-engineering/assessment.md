This quiz covers the whole Advanced LLM Engineering module: when to fine-tune, SFT and loss masking, LoRA and QLoRA memory budgets, preference data and Bradley-Terry, RLHF, DPO, quantisation, distillation, vLLM, speculative decoding and FlashAttention, continuous batching and KV caching. Aim for at least 13 out of 16 before moving on to Multimodal AI.

```quiz
[
  {
    "q": "A team wants their assistant to know about product prices that change daily. Which approach fits best?",
    "options": [
      "Fine-tune the model every night on the new prices",
      "Retrieve current prices at request time (RAG)",
      "Increase the model's temperature",
      "Distil the model into a smaller one"
    ],
    "answer": 1,
    "why": "Changing knowledge belongs in retrieval. Fine-tuning shapes behaviour and goes stale for facts."
  },
  {
    "q": "In SFT, what does this line do?\n\n`labels[:, :prompt_len] = -100`",
    "options": [
      "Deletes the prompt from the input",
      "Pads the prompt to a fixed length",
      "Sets the attention mask for the prompt to zero",
      "Masks the prompt tokens out of the loss so the model is trained only on the response"
    ],
    "answer": 3,
    "why": "`-100` is the default `ignore_index` of PyTorch cross-entropy. The model still conditions on the prompt but is not trained to produce it."
  },
  {
    "q": "After fine-tuning on narrow extraction data, the model's general instruction-following score drops. What is this called, and what is one mitigation?",
    "options": [
      "Catastrophic forgetting; mix general replay data into training or train less",
      "Reward hacking; raise beta",
      "Overfitting the tokenizer; retrain the tokenizer",
      "Quantisation error; use more bits"
    ],
    "answer": 0,
    "why": "Narrow training degrades unexercised abilities. Replay data, lower learning rates, fewer epochs and LoRA all reduce it."
  },
  {
    "q": "Roughly how much memory do weights, gradients and AdamW states need for full fine-tuning of a 7-billion-parameter model in mixed precision, excluding activations?",
    "options": [
      "About 14 GB",
      "About 28 GB",
      "About 112 GB",
      "About 1 TB"
    ],
    "answer": 2,
    "why": "About 16 bytes per trainable parameter: 7e9 x 16 = 112 GB."
  },
  {
    "q": "Why does QLoRA reduce memory compared with LoRA?",
    "options": [
      "It trains fewer adapter parameters",
      "It stores the frozen base weights in 4-bit NF4 instead of 16-bit",
      "It removes the optimiser",
      "It shortens the sequence length"
    ],
    "answer": 1,
    "why": "The frozen base dominates LoRA's memory; storing it in about half a byte per parameter cuts it by roughly four."
  },
  {
    "q": "Under the Bradley-Terry model, two responses have rewards 1.5 and 1.5. What is the probability the first is preferred?",
    "options": [
      "0.5",
      "0",
      "0.75",
      "1"
    ],
    "answer": 0,
    "why": "sigmoid(1.5 - 1.5) = sigmoid(0) = 0.5."
  },
  {
    "q": "In RLHF, what is the purpose of the KL penalty to the reference model?",
    "options": [
      "To speed up generation",
      "To train the reward model",
      "To reduce the vocabulary",
      "To keep the policy near the SFT model, where the reward model is reliable and language ability is preserved"
    ],
    "answer": 3,
    "why": "Without it, the policy drifts into regions where the reward model is wrong and exploits it."
  },
  {
    "q": "A PPO run shows reward-model scores rising steadily while independent judge win rates fall. What should you do?",
    "options": [
      "Train longer, since reward is rising",
      "Lower beta to allow more exploration",
      "Treat it as reward hacking: stop or roll back to an earlier checkpoint and investigate the reward model",
      "Increase the learning rate"
    ],
    "answer": 2,
    "why": "Divergence between proxy reward and independent evaluation is the signature of reward over-optimisation."
  },
  {
    "q": "Which statement about DPO is correct?",
    "options": [
      "Its loss is `-log sigmoid` of beta times the difference of policy-to-reference log-ratios for chosen and rejected responses",
      "It requires training a separate reward model first",
      "It needs a value model",
      "It samples new responses during every training step"
    ],
    "answer": 0,
    "why": "DPO writes the reward as beta times the log-ratio and fits Bradley-Terry directly; the partition function cancels."
  },
  {
    "q": "What does AWQ use calibration data for?",
    "options": [
      "To fine-tune the model before quantising",
      "To train a reward model",
      "To build the tokenizer",
      "To find input channels with large activations and choose per-channel scales that protect their weights"
    ],
    "answer": 3,
    "why": "AWQ scales salient channels up before quantisation (and the inputs down), with the scaling chosen to minimise output error on calibration inputs."
  },
  {
    "q": "In knowledge distillation, what does this compute?\n\n`F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction=\"batchmean\") * T * T`",
    "options": [
      "The cross-entropy between the student and the true labels",
      "The KL divergence from the teacher's softened distribution to the student's, scaled by T squared",
      "The teacher's accuracy",
      "The DPO loss"
    ],
    "answer": 1,
    "why": "It is KL(teacher || student) at temperature T, scaled by T^2 to keep gradient sizes stable across temperatures."
  },
  {
    "q": "How does PagedAttention increase the number of sequences vLLM can serve concurrently?",
    "options": [
      "By compressing the weights",
      "By shortening prompts",
      "By storing the KV cache in small fixed-size blocks allocated on demand, instead of reserving maximum length per request",
      "By running attention on the CPU"
    ],
    "answer": 2,
    "why": "On-demand blocks remove most of the memory that contiguous worst-case reservations waste."
  },
  {
    "q": "Speculative decoding with acceptance rate 0.8 and 4 draft tokens produces on average how many tokens per target forward pass?",
    "options": [
      "About 3.4",
      "About 1.8",
      "About 2.5",
      "Exactly 5"
    ],
    "answer": 0,
    "why": "(1 - 0.8^5) / (1 - 0.8) = 0.67232 / 0.2 = about 3.36."
  },
  {
    "q": "Which technique avoids writing the full attention matrix to GPU main memory while computing exact attention?",
    "options": [
      "Prefix caching",
      "GQA",
      "FlashAttention",
      "Speculative decoding"
    ],
    "answer": 2,
    "why": "FlashAttention tiles the computation in on-chip memory with an online softmax."
  },
  {
    "q": "Why does continuous batching achieve much higher throughput than static batching for LLM generation?",
    "options": [
      "It uses a smaller model",
      "It skips the prefill phase",
      "It generates several tokens per sequence per step",
      "Finished sequences leave and new ones join at every decode step, so slots do not sit idle waiting for the longest sequence"
    ],
    "answer": 3,
    "why": "Iteration-level scheduling removes the waste of slots held by finished sequences in a static batch."
  },
  {
    "q": "A model has 32 layers, 8 KV heads and head dimension 128, with a bf16 KV cache. How much cache does one 8,192-token sequence need?",
    "options": [
      "128 MiB",
      "1 GiB",
      "512 MiB",
      "4 GiB"
    ],
    "answer": 1,
    "why": "2 x 32 x 8 x 128 x 2 = 131,072 bytes per token (128 KiB); times 8,192 tokens = 1 GiB."
  }
]
```
