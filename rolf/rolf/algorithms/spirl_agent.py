from pathlib import Path
import copy
import torch

from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.policies.cl_model_policies import ACClModelPolicy
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.components.critic import SplitObsMLPCritic
from spirl.rl.agents.ac_agent import SACAgent
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.utils.general_utils import AttrDict, map_dict
from spirl.data.maze.src.maze_agents import MazeACActionPriorSACAgent
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.components.critic import MLPCritic
from spirl.models.skill_prior_mdl import SkillPriorMdl

from .base_agent import BaseAgent
from ..utils import Logger
from ..utils.pytorch import count_parameters


class SPiRLAgent(FixedIntervalHierarchicalAgent):
    def __init__(self, cfg, ob_space, ac_space):
        self._cfg = cfg
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._device = torch.device(cfg.device)
        self._buffer = None

        # set up configuration
        agent_config = self.setup_configs()
        agent_config.device = cfg.device
        FixedIntervalHierarchicalAgent.__init__(self, agent_config)

        self._log_creation()

    @property
    def ac_space(self):
        return self._ac_space

    def _log_creation(self):
        Logger.info("Creating a SPiRL agent")
        Logger.info(f"The hl agent has {count_parameters(self.hl_agent)} parameters")
        Logger.info(f"The ll agent has {count_parameters(self.ll_agent)} parameters")

    def setup_configs(self):
        if self._cfg.env == "maze":
            return self.maze_configs()
        elif self._cfg.env == "kitchen":
            return self.kitchen_configs()

    def maze_configs(self):
        from spirl.configs.default_data_configs.maze import data_spec

        # Replay Buffer
        replay_params = AttrDict(capacity=1e5, dump_replay=False)

        base_agent_params = AttrDict(
            batch_size=256,
            replay=UniformReplayBuffer,
            replay_params=replay_params,
            clip_q_target=False,
        )

        ###### Low-Level ######
        # LL Policy Model
        ll_model_params = AttrDict(
            state_dim=data_spec.state_dim,
            action_dim=data_spec.n_actions,
            n_rollout_steps=10,
            kl_div_weight=1e-2,
            prior_input_res=data_spec.res,
            n_input_frames=2,
            cond_decode=True,
        )

        # LL Policy
        ll_policy_params = AttrDict(
            policy_model=ImageClSPiRLMdl,
            policy_model_params=ll_model_params,
            policy_model_checkpoint=Path(
                "log/skill_prior_learning/maze/hierarchical_cl"
            ),
            initial_log_sigma=-50.0,
        )
        ll_policy_params.update(ll_model_params)

        # LL Critic
        ll_critic_params = AttrDict(
            action_dim=data_spec.n_actions,
            input_dim=data_spec.state_dim,
            output_dim=1,
            action_input=True,
            unused_obs_size=10,  # ignore HL policy z output in observation for LL critic
        )

        # LL Agent
        ll_agent_config = copy.deepcopy(base_agent_params)
        ll_agent_config.update(
            AttrDict(
                policy=ACClModelPolicy,
                policy_params=ll_policy_params,
                critic=SplitObsMLPCritic,
                critic_params=ll_critic_params,
            )
        )

        ###### High-Level ########
        # HL Policy
        hl_policy_params = AttrDict(
            action_dim=10,  # z-dimension of the skill VAE
            input_dim=data_spec.state_dim,
            max_action_range=2.0,  # prior is Gaussian with unit variance
            prior_model=ll_policy_params.policy_model,
            prior_model_params=ll_policy_params.policy_model_params,
            prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
        )

        # HL Critic
        hl_critic_params = AttrDict(
            action_dim=hl_policy_params.action_dim,
            input_dim=hl_policy_params.input_dim,
            output_dim=1,
            n_layers=2,  # number of policy network layers
            nz_mid=256,
            action_input=True,
            unused_obs_size=ll_model_params.prior_input_res**2
            * 3
            * ll_model_params.n_input_frames,
        )

        # HL Agent
        hl_agent_config = copy.deepcopy(base_agent_params)
        hl_agent_config.update(
            AttrDict(
                policy=ACLearnedPriorAugmentedPIPolicy,
                policy_params=hl_policy_params,
                critic=SplitObsMLPCritic,
                critic_params=hl_critic_params,
                td_schedule_params=AttrDict(p=1.0),
            )
        )

        ##### Joint Agent #######
        agent_config = AttrDict(
            hl_agent=MazeACActionPriorSACAgent,
            hl_agent_params=hl_agent_config,
            ll_agent=SACAgent,
            ll_agent_params=ll_agent_config,
            hl_interval=ll_model_params.n_rollout_steps,
            log_videos=False,
            update_hl=True,
            update_ll=False,
        )
        return agent_config

    def kitchen_configs(self):
        from spirl.configs.default_data_configs.kitchen import data_spec

        # Replay Buffer
        replay_params = AttrDict()

        base_agent_params = AttrDict(
            batch_size=256,
            replay=UniformReplayBuffer,
            replay_params=replay_params,
            clip_q_target=False,
        )

        ###### Low-Level ######
        # LL Policy
        ll_model_params = AttrDict(
            state_dim=data_spec.state_dim,
            action_dim=data_spec.n_actions,
            kl_div_weight=5e-4,
            nz_enc=128,
            nz_mid=128,
            n_processing_layers=5,
            nz_vae=10,
            n_rollout_steps=10,
            cond_decode=True,
        )

        # create LL closed-loop policy
        ll_policy_params = AttrDict(
            policy_model=ClSPiRLMdl,
            policy_model_params=ll_model_params,
            policy_model_checkpoint=Path(
                "log/skill_prior_learning/kitchen/hierarchical_cl"
            ),
        )
        ll_policy_params.update(ll_model_params)

        # LL Agent
        ll_agent_config = copy.deepcopy(base_agent_params)
        ll_agent_config.update(
            AttrDict(
                model=SkillPriorMdl,
                model_params=ll_model_params,
                model_checkpoint=Path(
                    "log/skill_prior_learning/kitchen/hierarchical_cl"
                ),
            )
        )

        ###### High-Level ########
        # HL Policy
        hl_policy_params = AttrDict(
            action_dim=ll_model_params.nz_vae,  # z-dimension of the skill VAE
            input_dim=data_spec.state_dim,
            max_action_range=2.0,  # prior is Gaussian with unit variance
            nz_mid=256,
            n_layers=5,
            prior_model=ll_policy_params.policy_model,
            prior_model_params=ll_policy_params.policy_model_params,
            prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
        )

        # HL Critic
        hl_critic_params = AttrDict(
            action_dim=hl_policy_params.action_dim,
            input_dim=hl_policy_params.input_dim,
            output_dim=1,
            n_layers=5,  # number of policy network laye
            nz_mid=256,
            action_input=True,
        )

        # HL Agent
        hl_agent_config = copy.deepcopy(base_agent_params)
        hl_agent_config.update(
            AttrDict(
                policy=LearnedPriorAugmentedPIPolicy,
                policy_params=hl_policy_params,
                critic=MLPCritic,
                critic_params=hl_critic_params,
                td_schedule_params=AttrDict(p=5.0),
            )
        )

        # create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
        ll_agent_config = AttrDict(
            policy=ClModelPolicy,
            policy_params=ll_policy_params,
            critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
            critic_params=hl_critic_params,
        )

        ##### Joint Agent #######
        agent_config = AttrDict(
            hl_agent=ActionPriorSACAgent,
            hl_agent_params=hl_agent_config,
            ll_agent=SACAgent,
            ll_agent_params=ll_agent_config,
            hl_interval=ll_model_params.n_rollout_steps,
            log_video_caption=True,
            update_ll=False,
        )

        return agent_config

    def is_off_policy(self):
        return True

    """ Dummy methods """

    def get_runner(self, cfg, env, env_eval):
        return

    def load_replay_buffer(self, replay_dir, ckpt_num):
        return
