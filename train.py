from game import Game
from model import select_action, optimize_model
from model import ReplayMemory, DQN
import torch
import torch.optim as optim
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(n_players=2, num_episodes=100, lr=1e-4, batch_size=128, gamma=0.99, tau=0.005,
                   eps_start=0.9, eps_end=0, eps_decay=1000):
    
    game = Game(n_players)
    memory = ReplayMemory(10000)
    
    scores = []
    episode_durations = []
    steps_done = 0
    
    n_actions = Game(n_players).n_actions
    n_observations = len(Game(n_players).encode())
    
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        #state, info = env.reset()
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    
        game = Game(n_players)
        state = torch.tensor(game.encode(), dtype=torch.float32, device=device).unsqueeze(0)
        
        t = 0
        done = False
        while not done:
            t += 1
            #print(state)
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
            action = select_action(state, policy_net, eps_threshold)
            steps_done += 1
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            observation, reward, terminated = game.play_AI(action)
            reward = torch.tensor([reward], device=device)
            done = terminated #or truncated
    
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma)
    
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
    
        episode_durations.append(t + 1)
        scores.append(sum(game.board.table.values()))
                
        if i_episode % 1000 == 999:
            print(i_episode)
    #         plot_durations()
    #         plt.plot([sum(scores[i:i+100]) for i in range(len(scores)-100)])
    #         plt.show()
    
    
    print('Complete')
    print('Average score:', sum(scores[-100:])/100)
    
    print('Average game length:', sum(episode_durations[-100:])/100)
    
    plt.plot([sum(scores[i:i+100])/100 for i in range(len(scores)-100)])
    plt.plot([sum(episode_durations[i:i+100])/100 for i in range(len(scores)-100)])