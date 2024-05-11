"""Define task initialization modes for FurnitureBench."""
from enum import Enum


class Randomness(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    MEDIUM_COLLECT = 3  # For data collection with medium-level perturbation.
    HIGH_COLLECT = 4
    SKILL_FIXED = 5
    SKILL_RANDOM = 6


def str_to_enum(v):
    if isinstance(v, Randomness):
        return v
    if v == "low":
        return Randomness.LOW
    elif v == "med":
        return Randomness.MEDIUM
    elif v == "high":
        return Randomness.HIGH
    elif v == "med_collect":
        return Randomness.MEDIUM_COLLECT
    elif v == "high_collect":
        return Randomness.HIGH_COLLECT
    elif v == "skill_fixed":
        return Randomness.SKILL_FIXED
    elif v == "skill_random":
        return Randomness.SKILL_RANDOM
    else:
        raise ValueError(f"Unknown initialization mode: {v}")

def load_embedding(encoder_type, device_id):
    if encoder_type == "r3m":
        from r3m import load_r3m

        img_emb_layer = load_r3m("resnet50").module
        embedding_dim = 2048
    elif encoder_type == "vip":
        from vip import load_vip

        model = load_vip().module
        embedding_dim = 1024
    elif encoder_type == "liv":
        from liv import load_liv

        model = load_liv().module
        embedding_dim = 1024
    elif encoder_type.startswith("clip"):
        import clip

        if encoder_type == "clip_vit_b16":
            model, _ = clip.load("ViT-B/16")
            embedding_dim = 512
        if encoder_type == "clip_vit_l14":
            model, _ = clip.load("ViT-L/14")
            embedding_dim = 768
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    model.requires_grad_(False)
    model.eval()
    model = model.to(device_id)

    if encoder_type.startswith("clip"):
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        def img_emb_layer(x):
            return model.encode_image(transform(x / 255.0))
    elif encoder_type in ["liv"]:

        def img_emb_layer(x):
            return model(x / 255.0)

    return img_emb_layer, embedding_dim
