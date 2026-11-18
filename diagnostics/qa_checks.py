"""QA diagnostics for Uni-Hema. Run inside Colab after `import app` (Step 7 of the notebook).

Usage in a notebook cell:
    import sys; sys.path.insert(0, "/content/Blood_Lense-AI/diagnostics")
    import qa_checks; qa_checks.run(app)
"""
from PIL import Image
import torch


def ask(app, path, q):
    img = Image.open(path).convert("RGB")
    t, _ = app.tarnsform_seg(img, None)
    with torch.no_grad():
        o = app.model(t.unsqueeze(0).to(app.device), [q], [{"completed_text": "cell A"}])
    return o["pred_text"][0]


def answers_per_image(app, q="Q: What can you observe in the image of the cell?"):
    """Same question on every repo QA sample: do the answers change with the image?"""
    print("-----", q)
    for path, _ in app.cloze_samples_QA:
        print(path.split("/")[-1], "->", ask(app, path, q))


def image_vs_question_effect(app, img_a, img_b,
                             q1="Q: What can you observe in the image of the cell?",
                             q2="Q: What is the texture of the cytoplasm?"):
    """Measure how much the decoder features change with the image vs with the question."""
    cap = {}

    def hook(m, i, o):
        x = o[0] if isinstance(o, (tuple, list)) else o
        cap["x"] = x.detach().float().cpu()

    h = app.model.xformer.register_forward_hook(hook)

    def emb(path, q):
        img = Image.open(path).convert("RGB")
        t, _ = app.tarnsform_seg(img, None)
        with torch.no_grad():
            app.model(t.unsqueeze(0).to(app.device), [q], [{"completed_text": "cell A"}])
        return cap["x"]

    base = emb(img_a, q1)
    print("scale             :", base.abs().mean().item())
    print("same q, other img :", (emb(img_b, q1) - base).abs().mean().item())
    print("same img, other q :", (emb(img_a, q2) - base).abs().mean().item())
    h.remove()


def run(app):
    answers_per_image(app)
    image_vs_question_effect(app, "sample_qa/EO_85775.jpg", "sample_qa/LY_810594.jpg")
