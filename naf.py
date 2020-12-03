import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
import random
import math

from collections import deque, namedtuple
import time
import gym
import copy

from torch.distributions import MultivariateNormal, Normal
import argparse
import wandb


class DeepNAF(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed):
        super(DeepNAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        
        concat_size = state_size+layer_size

        self.fc1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.fc2 = nn.Linear(concat_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.fc3 = nn.Linear(concat_size, layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)
        self.fc4 = nn.Linear(concat_size, layer_size)
        self.bn4 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
    
    def forward(self, input_, action=None):
        """
        
        """

        x = torch.relu(self.fc1(input_))
        x = self.bn1(x)
        x = torch.relu(self.fc2(torch.cat([x, input_],dim=1)))
        x = self.bn2(x)
        x = torch.relu(self.fc3(torch.cat([x, input_],dim=1)))
        x = self.bn3(x)
        x = torch.relu(self.fc4(torch.cat([x, input_],dim=1)))

        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        #dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        #wandb.log({"Action Noise": action.detach().cpu().numpy() - action_value.squeeze(-1).detach().cpu().numpy()})

        return action, Q, V

class NAF(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed):
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
                
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
        

    
    def forward(self, input_, action=None):
        """
        
        """

        x = torch.relu(self.head_1(input_))
        #x = self.bn1(x)
        x = torch.relu(self.ff_1(x))
        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        #dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        #wandb.log({"Action Noise": action.detach().cpu().numpy() - action_value.squeeze(-1).detach().cpu().numpy()})

        
        return action, Q, V
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, nstep):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = nstep
        self.n_step_buffer = deque(maxlen=nstep)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()

            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    
    def calc_multistep_return(self):
        Return = 0
        for idx in range(1):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, seed, gamma=0.99, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma = gamma

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        # n_step calc
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    
    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def __len__(self):
        return len(self.buffer)



class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 PER,
                 LR,
                 Nstep,
                 D2RL,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 NUPDATES,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.nstep = Nstep
        self.UPDATE_EVERY = UPDATE_EVERY
        self.NUPDATES = NUPDATES
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.per = PER

        self.action_step = 4
        self.last_action = None

        # Q-Network
        if D2RL == 0:
            self.qnetwork_local = NAF(state_size, action_size,layer_size, seed).to(device)
            self.qnetwork_target = NAF(state_size, action_size,layer_size, seed).to(device)
        else:
            self.qnetwork_local = DeepNAF(state_size, action_size,layer_size, seed).to(device)
            self.qnetwork_target = DeepNAF(state_size, action_size,layer_size, seed).to(device)
        
        wandb.watch(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        if PER == True:
            print("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, seed, n_step=self.nstep)
        else:
            print("Using Regular Experience Replay")
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, self.nstep)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0 
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                Q_losses = []
                for _ in range(self.NUPDATES):
                    experiences = self.memory.sample()
                    if self.per == True:
                        loss = self.learn_per(experiences)
                    else:
                        loss = self.learn(experiences)
                    self.Q_updates += 1
                    Q_losses.append(loss)
                #wandb.log({"Q_loss": np.mean(Q_losses), "Optimization step": self.Q_updates})


    def act(self, state):
        """Calculating the action
        
        Params
        ======
            state (array_like): current state
            
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _ = self.qnetwork_local(state.unsqueeze(0))
            #action = action.cpu().squeeze().numpy() + self.noise.sample()
            #action = np.clip(action, -1,1)[0]
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.action_size,))



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _ = self.qnetwork_local(states, actions)

        # Compute loss
        loss = F.mse_loss(Q, V_targets)
        
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        

        
        return loss.detach().cpu().numpy()

    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _ = self.qnetwork_local(states, actions)

        # Compute loss
        td_error = Q - V_targets# - Q
        loss = (td_error.pow(2)*weights).mean().to(self.device)
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update per priorities
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        
        return loss.detach().cpu().numpy()       

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


def evaluate(frame):
    scores = []
    with torch.no_grad():
        for i in range(Eval_runs):
            state = test_env.reset()    
            score = 0
            done = 0
            while not done:
                action = agent.act(state)
                state, reward, done, _ = test_env.step(action)
                score += reward
                if done:
                    scores.append(score)
                    break

    wandb.log({"Reward": np.mean(scores), "Step": frame})


def run(frames=1000):
    """"NAF.
    
    Params
    ======

    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    i_episode = 0
    state = env.reset()
    score = 0 
    evaluate(0)                 
    for frame in range(1, frames+1):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if frame % Eval_every == 0:
            evaluate(frame)

        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
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
    parser.add_argument('-n_up', "--n_updates", type=int, default=3,
                     help='update the network for x steps (default: 3)')
    parser.add_argument('-s', "--seed", type=int, default=0,
                     help='random seed (default: 0)')
    args = parser.parse_args()
    wandb.init(project="naf", name=args.info)
    wandb.config.update(args)
    seed = args.seed
    BUFFER_SIZE = args.mem
    per = args.per
    BATCH_SIZE = args.batch_size
    LAYER_SIZE = args.layer_size
    nstep = args.nstep
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.learning_rate
    UPDATE_EVERY = args.update_every
    NUPDATES = args.n_updates
    Eval_every = args.eval_every
    Eval_runs = args.eval_runs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    env = gym.make(args.env) #CartPoleConti
    test_env = gym.make(args.env)


    env.seed(seed)
    test_env.seed(seed+1)
    action_size     = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        layer_size=LAYER_SIZE,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE,
                        PER=per, 
                        LR=LR,
                        Nstep=nstep, 
                        D2RL=args.d2rl,
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY,
                        NUPDATES=NUPDATES,
                        device=device, 
                        seed=seed)



    t0 = time.time()
    run(frames = args.frames)
    t1 = time.time()
    
    print("Training time: {}min".format(round((t1-t0)/60,2)))
    torch.save(agent.qnetwork_local.state_dict(), "NAF_"+args.env+"_.pth")