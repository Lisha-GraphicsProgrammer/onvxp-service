# training_pipeline/validate_samples.py
"""
Visual pre-flight check for a candidate training dataset. Samples images
from the isolated class and asks a vision model directly: does this image
actually show the target class? Rejects the whole candidate dataset if
agreement is too low — BEFORE training ever starts, not after.
"""
import glob
import random
import base64
from pathlib import Path


def sample_images(dataset_dir: str, n: int = 12) -> list[str]:
    imgs = glob.glob(str(Path(dataset_dir) / "train" / "images" / "*"))
    random.shuffle(imgs)
    return imgs[:n]


def verify_with_dino(image_paths, class_prompt, confidence=0.3):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    results = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=class_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        out = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=confidence, text_threshold=confidence,
            target_sizes=[image.size[::-1]],
        )[0]
        results.append({"path": path, "found": len(out["boxes"]) > 0, "num_boxes": len(out["boxes"])})
    return results


def verify_with_claude_vision(image_paths, class_description, api_key):
    import requests

    results = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 20,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": f"Does this image clearly show {class_description}? Answer only yes or no."},
                    ],
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip().lower()
        results.append({"path": path, "found": text.startswith("y")})
    return results


def validate_dataset(dataset_dir: str, class_prompt: str, min_agreement: float = 0.6,
                      sample_n: int = 12, anthropic_api_key: str | None = None) -> dict:
    samples = sample_images(dataset_dir, sample_n)
    if not samples:
        return {"passed": False, "reason": "No images found to sample."}

    results = (
        verify_with_claude_vision(samples, class_prompt, anthropic_api_key)
        if anthropic_api_key else
        verify_with_dino(samples, class_prompt)
    )

    agree = sum(1 for r in results if r["found"]) / len(results)
    passed = agree >= min_agreement
    return {
        "passed": passed,
        "agreement_rate": round(agree, 2),
        "sample_size": len(results),
        "details": results,
        "reason": None if passed else
            f"Only {agree:.0%} of sampled images visually matched '{class_prompt}' (required {min_agreement:.0%})",
    }


if __name__ == "__main__":
    import sys, json
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/exposed_electrical_wire"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "exposed electrical wire"
    result = validate_dataset(dataset_dir, prompt)
    print(json.dumps(result, indent=2))