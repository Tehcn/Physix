import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

from os import system

if __name__ == '__main__':  # Required for multiprocessing

    system('clear') # clear console because yeah

    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    target_steps = 100_000
    agents_per_match = 2
    num_instances = 3
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps
    
    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            spawn_opponents=True,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=6.0,
                        save=30.0,
                        demo=7.0
                    )
                ),
                (
                    0.1, # VelocityPlayerToBallReward
                    1.0, # VelocityBallToGoalReward
                    1.0  # EventReward
                )
            ),
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Start a set amount of instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="auto",  # Need to set device again (if using a specific one)
            force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
        )
    except:
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn = Tanh,
            net_arch = [
                512, 512,
                dict(pi=[256, 256, 256], vf=[256, 256, 256])
            ]
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,                  # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )


    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    while True:
        try:
            mmr_steps = 25_000_000
            model.learn(mmr_steps, callback=callback)
            model.save("models/exit_save.zip")
            model.save(f"mmr_models/{model.num_timesteps}.zip")
        except KeyboardInterrupt:
            model.save("models/exit_save.zip")
            model.save(f"mmr_models/{model.num_timesteps}.zip")
            break