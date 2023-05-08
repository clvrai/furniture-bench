import time
import pickle
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import cv2
from joblib import Parallel, delayed
import gym

from furniture_bench.envs.furniture_env import FurnitureEnv
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv


class DataCollector:
    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW_RANDOM,
        gpu_id: int = 0,
        num_demos: int = 100,
    ):
        """Demonstration collection class.

        Args:
            is_sim: Whether to use simulator or real world environment.
            data_path: Path to save data.
            device_interface: Keyboard and/or Oculus interface.
            furniture: Name of the furniture.
            headless: Whether to use headless mode.
            draw_marker: Whether to draw AprilTag marker.
            manual_label: Whether to manually label the reward.
            scripted: Whether to use scripted function for getting action.
            randomness: Initialization mode.
            gpu_id: GPU ID.
            num_demos: The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
        """
        if is_sim:
            self.env = gym.make(
                "Furniture-Sim-Env-v0",
                furniture=furniture,
                headless=headless,
                num_envs=1, # Only support 1 for now.
                domain_randomization=False,
                resize_img=False,
                np_step_out=False, # Always output Tensor in this setting. Will change to numpy in this code.
                channel_first=False,
                randomness=randomness,
                compute_device_id=gpu_id,
                graphics_device_id=gpu_id,
                use_depth_cam=True,
                use_rear_cam=True,
            )
        else:
            self.env = gym.make(
                "Furniture-Env-v0",
                use_quat=True,
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )
        self.is_sim = is_sim
        self.data_path = data_path
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos

        self.scripted = scripted

        self.traj_counter = 1
        self._reset_states()

        self.num_success = 0
        self.num_fail = 0

    def collect(self):
        obs = self.env.reset()
        print("[data collection] Start collecting the data!")
        if not self.scripted:
            print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        done = False
        while True:
            if self.num_success >= self.num_demos:
                print(
                    f"Collected {self.num_demos}/{self.traj_counter} successful trajectories!"
                )
                break
            if self.scripted:
                action, skill_complete = self.env.get_assembly_action()
                collect_enum = CollectEnum.DONE_FALSE
            else:
                action, collect_enum = self.device_interface.get_action_from_input(
                    use_quat=True
                )
                skill_complete = int(collect_enum == CollectEnum.SKILL)
                if skill_complete == 1:
                    self.skill_set.append(skill_complete)

            if done or (
                collect_enum == CollectEnum.SUCCESS or collect_enum == CollectEnum.FAIL
            ):
                if self.is_sim:
                    for k, v in next_obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        else:
                            next_obs[k] = v.squeeze().cpu().numpy()

                self.org_obs.append(next_obs)

                n_ob = {}
                n_ob["color_image1"] = resize(next_obs["color_image1"])
                n_ob["color_image2"] = resize_crop(
                    next_obs["color_image2"], config["camera"]["color_img_size"]
                )
                n_ob["robot_state"] = next_obs["robot_state"]
                n_ob["parts_poses"] = next_obs["parts_poses"]
                self.obs.append(n_ob)

                if done and not self.env.furnitures[0].all_assembled():
                    print("Failed to assemble the furniture, reset without saving.")
                    obs = self.reset()
                    collect_enum = CollectEnum.SUCCESS
                    self.num_fail += 1
                else:
                    if done:
                        collect_enum = CollectEnum.SUCCESS

                    obs = self.save_and_reset(collect_enum, {})
                    self.num_success += 1
                self.traj_counter += 1
                print(f"Success: {self.num_success}, Fail: {self.num_fail}")
                done = False
                continue
            s = time.time()
            next_obs, rew, done, info = self.env.step(action)
            if done:
                skill_complete = 1
            if rew == 1:
                self.last_reward_idx = len(self.acts)

            self.whole.append(time.time() - s)
            if collect_enum == CollectEnum.REWARD:
                rew = self.env.furniture.manual_assemble_label(
                    self.device_interface.rew_key
                )
                if rew == 0:
                    # Correction.
                    self.rews[self.last_reward_idx] = 0
                    rew = 1

            self.step_counter += 1
            if not info["obs_success"]:
                print("Getting observation failed, save trajectory.")
                # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                self.rews.pop()
                self.acts.pop()
                obs = self.save_and_reset(CollectEnum.FAIL, info)
                continue
            print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} num assembled: {len(self.env.furniture.assembled_set)} Skill: {len(self.skill_set)}"
            )
            if info["action_success"]:
                if self.is_sim:
                    for k, v in obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        else:
                            obs[k] = v.squeeze().cpu().numpy()
                    rew = float(rew.squeeze().cpu())

                self.org_obs.append(obs.copy())
                ob = {}
                ob["color_image1"] = resize(obs["color_image1"])
                ob["color_image2"] = resize_crop(
                    obs["color_image2"], config["camera"]["color_img_size"]
                )
                ob["robot_state"] = obs["robot_state"]
                ob["parts_poses"] = obs["parts_poses"]
                self.obs.append(ob)
                self.acts.append(action)
                self.rews.append(rew)
                self.skills.append(skill_complete)
            obs = next_obs

    def save_and_reset(self, collect_enum: CollectEnum, info):
        # Save the collected data when it is CollectEnum.
        self.save(collect_enum, info)
        print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()

        self._reset_states()

        print("Start collecting the data!")
        if not self.scripted:
            print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        return obs

    def _reset_states(self):
        self.obs = []
        self.org_obs = []
        self.acts = []
        self.rews = []
        self.skills = []
        self.whole = []
        self.step_counter = 0
        self.last_reward_idx = -1
        self.skill_set = []

    def save(self, collect_enum: CollectEnum, info):
        print(f"Length of trajectory: {len(self.obs)} ")

        data_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        os.makedirs(osp.join(self.data_path, data_name))

        # Color.
        self.color_names = ["color_image1", "color_image2", "color_image3"]
        self.color_video_names = []
        for name in self.color_names:
            self.color_video_names.append(
                Path(
                    osp.join(self.data_path, data_name, data_name + f"_{name}" + ".mp4")
                )
            )
        # Depth.
        self.depth_names = ["depth_image1", "depth_image2", "depth_image3"]
        self.depth_paths = []
        for name in self.depth_names:
            self.depth_paths.append(
                Path(osp.join(self.data_path, data_name, data_name + f"_{name}"))
            )

        path = Path(osp.join(self.data_path, data_name, data_name + ".pkl"))
        with open(path, "wb") as f:
            data = {}
            data["observations"] = self.obs
            data["actions"] = self.acts
            data["rewards"] = self.rews
            data["skills"] = self.skills
            data["success"] = True if collect_enum == CollectEnum.SUCCESS else False
            data["furniture"] = self.furniture

            if "error" in info:
                data["error_description"] = info["error"].value
                data["error"] = True
            else:
                data["error"] = False
                data["error_description"] = ""

            if not self.is_sim:
                data["cam2_to_base"] = self.env.cam2_to_base
                data["cam3_to_base"] = self.env.cam3_to_base

                data["cam1_intr"] = self.env.cam_intrs[0]
                data["cam2_intr"] = self.env.cam_intrs[1]
                data["cam3_intr"] = self.env.cam_intrs[2]

            # Save raw color images in mp4.
            print("Start saving raw color images.")
            outs = []
            s = time.time()
            for n in self.color_video_names:
                outs.append(
                    cv2.VideoWriter(
                        str(n),
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        10,
                        config["camera"]["color_img_size"],
                    )
                )
            for i, k in enumerate(self.color_names):
                for obs in self.org_obs:
                    outs[i].write(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR))
                outs[i].release()

            print(time.time() - s)
            # Save raw depth images in png.
            print("Start saving raw depth images.")
            s = time.time()
            for i, k in enumerate(self.depth_names):
                os.makedirs(self.depth_paths[i])
                Parallel(n_jobs=8)(
                    delayed(cv2.imwrite)(
                        f"{self.depth_paths[i]}/{j:05}.png",
                        obs[k],
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 5],
                    )
                    for j, obs in enumerate(self.org_obs)
                )
            print(time.time() - s)
            pickle.dump(data, f)

        print(f"Data saved at {path}")
