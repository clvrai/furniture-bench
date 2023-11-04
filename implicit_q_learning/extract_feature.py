import pickle
from pathlib import Path

import furniture_bench

import numpy as np
import torch
import gym
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string("furniture", None, "Furniture name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_file_path", None, "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", False, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", False, "Use vip to encode images.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_demos", None, "Number of demos to convert")
flags.DEFINE_integer('batch_size', 512, 'Batch size for encoding images')


def main(_):
    if FLAGS.num_threads > 0:
        print(f"Setting torch.num_threads to {FLAGS.num_threads}")
        torch.set_num_threads(FLAGS.num_threads)

    env_type = 'Image'
    env_id = f"Furniture-{env_type}-Dummy-v0"
    furniture = FLAGS.furniture
    demo_dir = FLAGS.demo_dir

    dir_path = Path(demo_dir)

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    if FLAGS.use_r3m:
        # Use R3M for the image encoder.
        from r3m import load_r3m
        encoder = load_r3m('resnet50')

    if FLAGS.use_vip:
        # Use VIP for the image encoder.
        from vip import load_vip
        encoder = load_vip()

    if FLAGS.use_r3m or FLAGS.use_vip:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to('cuda')
        device = torch.device("cuda")

    files = list(dir_path.glob('*.pkl'))
    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    for i, file_path in enumerate(files):
        if FLAGS.num_demos and i == FLAGS.num_demos:
            break
        print(f"Loading [{i+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)

            if len(x['observations']) == len(x['actions']):
                # Dummy
                x['observations'].append(x['observations'][-1])
            l = len(x["observations"])

            if FLAGS.use_r3m or FLAGS.use_vip:
                img1 = [x["observations"][i]["color_image1"] for i in range(l)]
                img2 = [x["observations"][i]["color_image2"] for i in range(l)]
                img1 = torch.from_numpy(np.stack(img1))
                img2 = torch.from_numpy(np.stack(img2))

                feature_dim = 2048 if FLAGS.use_r3m else 1024 
                img1_feature = np.zeros((l, feature_dim), dtype=np.float32)
                img2_feature = np.zeros((l, feature_dim), dtype=np.float32)

                with torch.no_grad():
                    # Use batch size.
                    for i in range(0, l, FLAGS.batch_size):
                        img1_feature[i:i+FLAGS.batch_size] = encoder(img1[i:i+FLAGS.batch_size].to(device).reshape(-1, 3, 224, 224)).cpu().detach().numpy()
                        img2_feature[i:i+FLAGS.batch_size] = encoder(img2[i:i+FLAGS.batch_size].to(device).reshape(-1, 3, 224, 224)).cpu().detach().numpy()

            for i in range(l - 1):
                if FLAGS.use_r3m or FLAGS.use_vip:
                    image1 = img1_feature[i]
                    next_image1 = img1_feature[i + 1]
                    image2 = img2_feature[i]
                    next_image2 = img2_feature[i + 1]
                else:
                    image1 = np.moveaxis(x['observations'][i]['color_image1'], 0, -1)
                    next_image1 = np.moveaxis(x['observations'][i + 1]['color_image1'], 0, -1)
                    image2 = np.moveaxis(x['observations'][i]['color_image2'], 0, -1)
                    next_image2 = np.moveaxis(x['observations'][i + 1]['color_image2'], 0, -1)

                obs_.append({
                    # 'image_feature': feature1,
                    'image1': image1,
                    'image2': image2,
                    'robot_state': x["observations"][i]['robot_state']
                })
                next_obs_.append({
                    # 'image_feature': next_feature1,
                    'image1': next_image1,
                    'image2': next_image2,
                    'robot_state': x["observations"][i + 1]['robot_state']
                })

                action_.append(x["actions"][i])
                reward_.append(x["rewards"][i])
                done_.append(1 if i == l - 2 else 0)

    dataset = {
        "observations": obs_,
        "actions": np.array(action_),
        "next_observations": next_obs_,
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }

    path = f'data/{env_type}/{furniture}.pkl' if FLAGS.out_file_path is None else FLAGS.out_file_path
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with Path(path).open("wb") as f:
        pickle.dump(dataset, f)
        print(f"Saved at {path}")


if __name__ == "__main__":
    app.run(main)
