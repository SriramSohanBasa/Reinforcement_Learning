import argparse
import os
from distutils.util import strtobool
import random
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import subprocess
import gym 
import pygame
from pygame.locals import *
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        default=os.path.basename(__file__).rstrip(".py"),
                        help='proximal policy')
    parser.add_argument('--gym-id',
                        type=str,
                        default="CartPole-v1",
                        help='the id if the gym environment')
    parser.add_argument('--learning-rate', type=float,
                        default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps',
                        type=int,
                        default=25000,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic',
                        type=lambda x: bool(strtobool(x)),
                        default=True, nargs='?', const=True,
                        help='if toggled,`torch.backends.cudnn.deterministic=False` ')
    parser.add_argument('--cuda',
                        type=lambda x: bool(strtobool(x)),
                        default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', 
                        type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, we will track the experiment with Weights and Biases')
    parser.add_argument('--wandb-project-name',
                        type=str,
                        default="cleanRL",
                        help='the wandb project name')
    parser.add_argument('--wandb-entity',
                        type=str,
                        default=None,
                        help='the entity (team) name that this experiment belongs to')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help="whether to capture videos of the agent's performances (check out 'video' folder)")
    parser.add_argument('--num-envs', type=int, default=4,
                        help='the number of parallel game environments')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    #print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    for i in range(100):
        writer.add_scalar("test_loss", i * 2, global_step=i)

    try:
        subprocess.Popen(["tensorboard", "--logdir", log_dir])
    except FileNotFoundError:
        print("TensorBoard not found. Make sure it is installed and added to your PATH.")

    
    ##seeding
    random.seed(args.seed) ##seeding means that we are setting the random number generator to a specific value so that we can get the same random numbers every time we run the code

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ##ppo deals with vector env , which stacks multiple independent env 

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx==0:
                    env= gym.wrappers.RecordVideo(env,
                    f"videos/{run_name}", record_video_trigger=lambda x: x%100 == 0)
                    env.seed(seed)
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)
            return env
        return thunk




    ##env setup
    envs= gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed+i,i, args.capture_video, run_name)
        for i in range(args.num_envs)]
    ) ##tunning the n creating functions 
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"
    print("envs.single_action_space", envs.single_action_space.n)
    print("envs.single_observation_space", envs.single_observation_space.shape)