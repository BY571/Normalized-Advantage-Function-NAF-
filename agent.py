from replay_buffer import ReplayBuffer, PrioritizedReplay   
from networks import NAF, DeepNAF

import torch
import torch.nn as nn 
from torch.nn.utils import clip_grad_norm_
import numpy as np 
import torch.optim as optim
import random

class NAF_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 args,
                 writer):
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
        self.seed = random.seed(args.seed)
        self.device = device
        self.TAU = args.tau
        self.GAMMA = args.gamma
        self.nstep = args.nstep
        self.UPDATE_EVERY = args.update_every
        self.NUPDATES = args.n_updates
        self.BATCH_SIZE = args.batch_size
        self.Q_updates = 0
        self.per = args.per
        self.clip_grad = args.clip_grad
        
        self.action_step = 4
        self.last_action = None

        # Q-Network
        if args.d2rl == 0:
            self.qnetwork_local = NAF(state_size, action_size, args.layer_size, args.seed).to(device)
            self.qnetwork_target = NAF(state_size, action_size, args.layer_size, args.seed).to(device)
        else:
            self.qnetwork_local = DeepNAF(state_size, action_size, args.layer_size, args.seed).to(device)
            self.qnetwork_target = DeepNAF(state_size, action_size, args.layer_size, args.seed).to(device)
        
        #wandb.watch(self.qnetwork_local)
        self.writer = writer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.learning_rate)
        print(self.qnetwork_local)
        
        # Replay memory
        if args.per == True:
            print("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplay(buffer_size=args.mem, 
                                            batch_size=args.batch_size,
                                            seed=args.seed,
                                            gamma=args.gamma,
                                            n_step=self.nstep,
                                            beta_frames=args.frames)
        else:
            print("Using Regular Experience Replay")
            self.memory = ReplayBuffer(buffer_size=args.mem,
                                       batch_size=args.batch_size,
                                       device=self.device,
                                       seed=args.seed,
                                       gamma=args.gamma,
                                       nstep=args.nstep)
        
        # define loss
        if args.loss == "mse":
            self.loss = nn.MSELoss()
        elif args.loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            print("Loss is not defined choose between mse and huber!")
        
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
                self.writer.add_scalar("Q_loss", np.mean(Q_losses), self.Q_updates)
                #.log({"Q_loss": np.mean(Q_losses), "Optimization step": self.Q_updates})

    def act_without_noise(self, state):
        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            _, _, _, action = self.qnetwork_local(state.unsqueeze(0))
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.action_size,))

    def act(self, state):
        """Calculating the action
        
        Params
        ======
            state (array_like): current state
            
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _, _ = self.qnetwork_local(state.unsqueeze(0))
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.action_size,))



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """

        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_, _ = self.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _, _ = self.qnetwork_local(states, actions)

        # Compute loss
        loss = self.loss(Q, V_targets) 
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad)
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
            _, _, V_, _ = self.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _, _ = self.qnetwork_local(states, actions)

        # Compute loss
        td_error = Q - V_targets
        loss = (self.loss(Q, V_targets)*weights).mean().to(self.device)
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad)
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
