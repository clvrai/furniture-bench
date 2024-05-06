import h5py
import argparse
import os
import numpy as np

# import torchvision.transforms as T
from torchvision.transforms import v2


import torch
from tqdm import tqdm
from PIL import Image


def feature_extractor_factory(feature_extractor):
    if feature_extractor == "r3m":
        from r3m import load_r3m

        r3m = load_r3m("resnet50")  # resnet18, resnet34
        r3m = r3m.cuda()
        r3m.eval()

        # transforms = v2.Compose(
        #     [v2.Resize(256), v2.CenterCrop(224)]
        # )  # ToTensor() divides by 255
        transforms = lambda x: x

        return r3m, transforms
    elif feature_extractor == "liv":
        from liv import load_liv

        liv = load_liv()
        liv = liv.cuda()
        liv.eval()
        transforms = lambda x: x.float() / 255.0
        return liv, transforms
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="directory to hdf5 dataset")
    parser.add_argument("--dataset", type=str, help="dataset name")

    parser.add_argument(
        "--feature-extractor", type=str, required=True, choices=["r3m", "liv"]
    )
    args = parser.parse_args()

    device = "cuda"
    input_files = []
    if args.dataset:
        input_files.append(args.dataset)
    else:  # Read from directory
        for root, dirs, files in os.walk(args.data_dir):
            for f in files:
                if f.endswith(".hdf5"):
                    input_files.append(os.path.join(root, f))

    feature_extractor, transforms = feature_extractor_factory(args.feature_extractor)

    batch_size = 512
    for dataset in input_files:
        with h5py.File(dataset, "a") as f:
            demos = sorted(list(f["data"].keys()), key=lambda x: int(x[5:]))
            for ep in tqdm(demos):
                for k in f["data/{}".format(ep)]:
                    if k in ["obs", "next_obs"]:
                        for obs_k in f["data/{}/{}".format(ep, k)]:
                            if obs_k in ["agentview_image", "robot0_eye_in_hand_image"]:
                                data_path = "data/{}/{}/{}".format(ep, k, obs_k)
                                temp_data_path = data_path + "_temp"
                                input_img = f[data_path][...]
                                input_img = np.transpose(
                                    input_img, (0, 3, 1, 2)
                                )  # Channel first.

                                all_embeddings = (
                                    []
                                )  # List to accumulate batch embeddings

                                # Process in minibatches
                                for i in range(0, len(input_img), batch_size):
                                    batch_imgs = input_img[i : i + batch_size]
                                    preprocessed_batch = transforms(
                                        torch.from_numpy(batch_imgs.astype(np.uint8))
                                    )
                                    preprocessed_batch = preprocessed_batch.to(device)

                                    with torch.no_grad():
                                        batch_embedding = (
                                            feature_extractor(preprocessed_batch)
                                            .cpu()
                                            .numpy()
                                        )
                                        all_embeddings.append(batch_embedding)

                                # Concatenate all batch embeddings
                                final_embedding = np.concatenate(all_embeddings, axis=0)

                                # Create the final dataset
                                f.create_dataset(temp_data_path, data=final_embedding)

                                # del f[data_path]

                                # Add _feature
                                data_path = (
                                    data_path + f"_{args.feature_extractor}_feature"
                                )

                                f.create_dataset(data_path, data=f[temp_data_path][...])

                                del f[temp_data_path]
