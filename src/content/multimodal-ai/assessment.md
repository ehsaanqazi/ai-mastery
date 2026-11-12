This quiz covers the whole Multimodal AI module: image understanding, vision-language models, speech recognition, text-to-speech, audio processing, video understanding, multimodal embeddings and multimodal RAG. Aim for at least 13 out of 16 before moving on to AI Security and Responsible AI.

```quiz
[
  {
    "q": "You need exact bounding boxes and a stable set of 20 defect classes on a production line running at high frame rates. What is the most suitable tool?",
    "options": [
      "A general-purpose vision-language model prompted per image",
      "A trained object detector",
      "A CLIP-style dual encoder with text prompts",
      "An OCR engine"
    ],
    "answer": 1,
    "why": "Classic detectors are fast, give precise coordinates and handle a fixed label set well; VLMs are slower and weak at localisation."
  },
  {
    "q": "In an OpenAI-style chat request, where does a base64-encoded image go?",
    "options": [
      "In a `data:` URL inside an image content part of the message's `content` list",
      "In a separate `images` parameter of the request",
      "Appended to the system prompt as plain text",
      "In an HTTP header"
    ],
    "answer": 0,
    "why": "Multimodal messages hold a list of typed parts; an image part carries either an HTTPS URL or a `data:` URL with the base64 bytes."
  },
  {
    "q": "A projector-based VLM uses 14-pixel patches and no patch merging. How many image tokens does a 448 by 448 image produce?",
    "options": ["256", "576", "1024", "3136"],
    "answer": 2,
    "why": "(448 / 14) x (448 / 14) = 32 x 32 = 1024 patches, each becoming one token."
  },
  {
    "q": "During the first (alignment) stage of training a LLaVA-style VLM, which component is usually trained?",
    "options": [
      "Only the vision encoder",
      "Only the LLM",
      "Only the projector",
      "Everything, from scratch"
    ],
    "answer": 2,
    "why": "The pretrained encoder and LLM are frozen while the small projector learns to map patch embeddings into the LLM's space."
  },
  {
    "q": "Which limitation of VLMs is best addressed by running a dedicated detector and passing its output to the model as text?",
    "options": [
      "Writing style",
      "Counting objects accurately",
      "Following instructions",
      "Translating languages"
    ],
    "answer": 1,
    "why": "VLMs summarise patches rather than enumerate objects, so counts are unreliable; a detector counts explicitly."
  },
  {
    "q": "Reference: 'set an alarm for seven' (5 words). Hypothesis: 'set the alarm for seven thirty'. What is the WER?",
    "options": ["0.2", "0.4", "0.33", "0.6"],
    "answer": 1,
    "why": "One substitution ('an' to 'the') and one insertion ('thirty'): 2 / 5 = 0.4."
  },
  {
    "q": "In this edit-distance recurrence, what does the term `d[i-1][j] + 1` represent?\n\n`d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)`",
    "options": [
      "An insertion",
      "A substitution",
      "A deletion of a reference word",
      "A match"
    ],
    "answer": 2,
    "why": "Moving from row i-1 to row i consumes a reference word without consuming a hypothesis word: a deletion."
  },
  {
    "q": "Why can two vendors' WER figures on similar-sounding test sets not be compared directly?",
    "options": [
      "WER is always above 1",
      "Test audio and text normalisation (numbers, punctuation, contractions) differ",
      "WER only works for English",
      "WER ignores substitutions"
    ],
    "answer": 1,
    "why": "Normalisation choices and audio conditions can change WER more than the model does; compare on your own data with one normaliser."
  },
  {
    "q": "In a two-stage neural TTS system, which component turns a mel spectrogram into audio samples?",
    "options": ["The text normaliser", "The acoustic model", "The vocoder", "The phonemiser"],
    "answer": 2,
    "why": "The acoustic model predicts the spectrogram; the vocoder renders the waveform."
  },
  {
    "q": "A team wants to offer custom cloned voices. Which practice is essential?",
    "options": [
      "Verified, informed consent from the voice owner, scoped to the intended use and revocable",
      "Cloning only voices found in public videos",
      "Keeping generated audio unlabelled so it sounds natural",
      "Storing voice samples indefinitely for future products"
    ],
    "answer": 0,
    "why": "Voice is biometric and cloning enables fraud; consent, scope, revocation, disclosure and provenance are the baseline."
  },
  {
    "q": "A 6 kHz tone is sampled at 8 kHz without an anti-aliasing filter. At what frequency does it appear?",
    "options": ["6 kHz", "4 kHz", "2 kHz", "14 kHz"],
    "answer": 2,
    "why": "Above the 4 kHz Nyquist limit, it folds back to 8 - 6 = 2 kHz."
  },
  {
    "q": "With a 16 kHz sampling rate, a 400-sample window and a 512-point FFT, how many frequency bins does each STFT frame have?",
    "options": ["200", "257", "400", "512"],
    "answer": 1,
    "why": "A real FFT of length 512 returns 512 / 2 + 1 = 257 bins from 0 Hz to the Nyquist frequency."
  },
  {
    "q": "You must answer 'When does the speaker first show the architecture diagram?' about a one-hour talk within a modest token budget. What is the best approach?",
    "options": [
      "Send 16 uniformly spaced frames to a VLM",
      "Send every frame at full resolution",
      "Use the timestamped transcript and scene changes to find candidate windows, then send dense frames from those windows",
      "Average all frame embeddings and compare with the question"
    ],
    "answer": 2,
    "why": "Sparse uniform frames will likely miss the moment; a cheap first pass localises candidates, then dense frames confirm."
  },
  {
    "q": "In the CLIP loss below, what is the role of `targets`?\n\n`logits = img @ txt.T / t; targets = torch.arange(N); loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2`",
    "options": [
      "They store the class labels of the images",
      "They mark that row i's correct match is column i, the diagonal of the similarity matrix",
      "They set the temperature",
      "They select which images to drop from the batch"
    ],
    "answer": 1,
    "why": "Each image's own caption sits on the diagonal; cross-entropy pushes each row and column to pick it."
  },
  {
    "q": "In a mixed index of CLIP image and text embeddings, text queries keep returning only text items. What is the likely cause?",
    "options": [
      "The modality gap between image and text embeddings",
      "The index uses too few dimensions",
      "Images were not normalised to grayscale",
      "The temperature was too low during indexing"
    ],
    "answer": 0,
    "why": "Image and text vectors occupy different regions, biasing raw similarities; search per modality and fuse the results."
  },
  {
    "q": "Why does MaxSim late-interaction retrieval find a page whose only relevant content is a small chart surrounded by boilerplate?",
    "options": [
      "It averages all patch vectors before comparing",
      "Each query token is scored by its best-matching patch, so irrelevant patches do not dilute the score",
      "It runs OCR and keyword search on the chart",
      "It retrieves only pages with images"
    ],
    "answer": 1,
    "why": "Taking the maximum per query token ignores boilerplate patches; single-vector averaging blends the chart into them."
  }
]
```
