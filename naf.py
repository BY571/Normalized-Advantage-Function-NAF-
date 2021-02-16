import torch
import numpy as np

import random

from collections import deque
import time
import gym
import pybullet_envs

import argparse
#import wandb
from torch.utils.tensorboard import SummaryWriter
from agent import NAF_Agent


def evaluate(frame, eval_runs):
    scores = []
    with torch.no_grad():
        for i in range(eval_runs):
            state = test_env.reset()    
            score = 0
            done = 0
            while not done:
                action = agent.act_without_noise(state)
                state, reward, done, _ = test_env.step(action)
                score += reward
                if done:
                    scores.append(score)
                    break

    #wandb.log({"Reward": np.mean(scores), "Step": frame})
    writer.add_scalar("Reward", np.mean(scores), frame)
    
def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def run(args):
    """"NAF.
    
    Params
    ======

    """
    frames = args.frames
    eval_every = args.eval_every
    eval_runs = args.eval_runs
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    i_episode = 0
    state = env.reset()
    score = 0 
    evaluate(0, eval_runs)                 
    for frame in range(1, frames+1):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if frame % eval_every == 0:
            evaluate(frame, eval_runs)

        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode, frame, frames, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode,frame, frames, np.mean(scores_window)))
            i_episode +=1 
            state = env.reset()
            score = 0              



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-info", type=str, default="Experiment-1",
                     help="Name of the Experiment (default: Experiment-1)")
    parser.add_argument('-env', type=str, default="Pendulum-v0",
                     help='Name of the environment (default: Pendulum-v0)')
    parser.add_argument('-f', "--frames", type=int, default=40000,
                     help='Number of training frames (default: 40000)')    
    parser.add_argument("--eval_every", type=int, default=1000,
                     help="Evaluate the current policy every X steps (default: 1000)")
    parser.add_argument("--eval_runs", type=int, default=3,
                     help="Number of evaluation runs to evaluate - averating the evaluation Performance over all runs (default: 3)")
    parser.add_argument('-mem', type=int, default=100000,
                     help='Replay buffer size (default: 100000)')
    parser.add_argument('-per', type=int, choices=[0,1],  default=0,
                     help='Use prioritized experience replay (default: False)')
    parser.add_argument('-b', "--batch_size", type=int, default=128,
                     help='Batch size (default: 128)')
    parser.add_argument('-nstep', type=int, default=1,
                     help='nstep_bootstrapping (default: 1)')
    parser.add_argument("-d2rl", type=int, choices=[0,1], default=0,
                     help="Using D2RL Deep Dense NN Architecture if set to 1 (default: 0)")
    parser.add_argument('-l', "--layer_size", type=int, default=256,
                     help='Neural Network layer size (default: 256)')
    parser.add_argument('-g', "--gamma", type=float, default=0.99,
                     help='Discount factor gamma (default: 0.99)')
    parser.add_argument('-t', "--tau", type=float, default=0.01,
                     help='Soft update factor tau (default: 0.01)')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                     help='Learning rate (default: 1e-3)')
    parser.add_argument('-u', "--update_every", type=int, default=1,
                     help='update the network every x step (default: 1)')
    parser.add_argument('-n_up', "--n_updates", type=int, default=1,
                     help='update the network for x steps (default: 1)')
    parser.add_argument('-s', "--seed", type=int, default=0,
                     help='random seed (default: 0)')
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients (default: 1.0)")
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse", help="Choose loss type MSE or Huber loss (default: mse)")
    
    args = parser.parse_args()
    #wandb.init(project="naf", name=args.info)
    #wandb.config.update(args)
    writer = SummaryWriter("runs/"+args.info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    
    env = gym.make(args.env) #CartPoleConti
    test_env = gym.make(args.env)

    seed = args.seed
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed+1)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = NAF_Agent(state_size=state_size,
                      action_size=action_size,
                      device=device, 
                      args= args,
                      writer=writer)



    t0 = time.time()
    run(args)
    t1 = time.time()
    
    timer(t0, t1)
    torch.save(agent.qnetwork_local.state_dict(), "NAF_"+args.info+"_.pth")
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)