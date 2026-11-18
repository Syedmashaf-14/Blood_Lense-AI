# Blood Lense AI: Uni-Hema on Google Colab

Notebook and notes for running the **Uni-Hema** hematology model (CVPR 2026) in Google Colab,
with fixes for current PyTorch/Transformers and a look at why Question Answering ignores the image.

This work builds on the original project. Please credit and check the license of the original code:

- Repo: https://github.com/intelligentMachines-ITU/Uni-Hema
- Paper: https://arxiv.org/abs/2511.13889
- Weights: https://huggingface.co/ryhm/Uni-hema

This repo does **not** contain the original code or the model weights. The notebook clones the original repo and downloads the weights.

## How to run

1. Open `Blood_Lense.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. Run the cells in order: Step 1, 2, 3, 4, **4b**, 6, 7, 8, 9. Step 5 (CUDA compile) is optional.
3. If you clone the original repo again, run Step 4 and Step 4b again.

## What the notebook fixes

- `app.py`: `torch.load(..., weights_only=False)` (PyTorch 2.6+), a fallback font, list output from QA/MLM shown as text, no auto-launch on import.
- `app.py`: the QA question box was locked (`interactive=False`); it is now editable.
- `models/dino/dino.py`: `generate()` only received the question tokens, so QA answers ignored the image. It now receives the image-conditioned encoder output (Step 4b).
- Pure-PyTorch fallback for deformable attention when the CUDA operator is not compiled.

## Known limitation: QA depends mostly on the question

We measured the features that go into the text decoder (`diagnostics/qa_checks.py`):

| Change | Mean absolute difference |
|---|---|
| Scale of the features | 0.114 |
| Same question, different image | 0.0016 |
| Same image, different question | 0.132 |

The image changes the features by about 1.4% of their scale, the question by more than 100%. After the Step 4b fix some answers change with the image, but many stay the same. We have not proven why; it looks like a model limitation rather than a code bug.

Detection, Classification, Morphology text and Segmentation did change with the image in our tests, so they are the reliable parts. We only checked that outputs change, not how accurate they are.

## Idea in progress

A RAG chatbot: run detection and classification on the image, turn the results into a findings text, retrieve hematology reference passages, and let a language model answer from that context only. This is decision support, not a diagnosis.

## Files

- `Blood_Lense.ipynb`: the Colab notebook with all patches.
- `diagnostics/qa_checks.py`: QA checks used for the analysis above.
