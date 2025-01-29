"""
    Let AI play the game.
    Load pytorch model and correct params.
    Run the game.
"""
from game_env import Environment
import torch
from dqn_module import DQNagent
from dataclass_helper import Args
from ppo_module import Memory
from ppo_module import PPOagent

if __name__ == "__main__":

    device = "cpu"
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005
    LR = 1e-3
    EPISODES = 4000
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 1000
    N_ACTIONS = 4
    N_OBSERVATIONS = 9
    current_game = True

    args = Args(
    batch_size=[5000],
    gamma=[0.99],
    tau=[0.005],
    lr=[1e-3],
    episodes=[6000],
    eps_start=[0.9],
    eps_end=[0.1],
    eps_decay=[1000],
    n_actions=[4],
    n_observations=[9],
    game_map="track 1",
    architecture="2 layer 64",
    loss_function="",
    location="random",
    start_pos="static",
    minibatch_size= 1250,
    num_iterations= 4000,
    seed= 1,
    num_steps= [5000],
    num_envs= 1,
    update_epochs= [4],
    clip_coef= [0.2],
    clip_vloss= True,
    learning_rate=2.5e-4,
    anneal_lr= True,
    gae_lambda= 0.95,
    norm_adv= True,
    ent_coef= 0.01,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    target_kl= None
)

    ppodevice = torch.device("cpu")
    memory = Memory(args.num_steps[0], args.num_envs, ppodevice)
    
    ppo_agent = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.learning_rate).to(ppodevice)
    dqn_agent2 = DQNagent(device, BATCH_SIZE, N_OBSERVATIONS, N_ACTIONS, GAMMA, EPS_END, EPS_START, EPS_DECAY, LR, TAU)
    
    file_name = "saved_models/ppo_64_static_track 1234_86.64.pth"
    file_name2 = "saved_models/track 1 128/someusefulname_64_randomspawn_track1239_52.07.pth"

    ppo_agent.load(file_name)
    dqn_agent2.load(file_name2)
    
    all_agents = [ppo_agent, dqn_agent2]

    MAP = "track 1"
    START_POS = "static"
    NUMBER_OF_PLAYERS = 2
    env = Environment("player", MAP, START_POS, NUMBER_OF_PLAYERS)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    apwojd = 0

    while current_game:

        #TODO: Clean this up
        if apwojd == 0:
            all_states = [state, state]
        apwojd = 1

        all_actions = []
        
        for i, agent in enumerate(all_agents):
            print(i)
            if i == 1:
                action = agent.select_action(all_states[i][0], env, False)
                all_actions.append(action.item())
            
            elif i == 0:
                action, _, _, _ = agent.get_action_and_value(all_states[i][0])
                all_actions.append(action)

        all_car_data = env.step(all_actions)

        all_states = []
        for car_data in all_car_data:
            
            observation, reward, terminated, truncated = car_data
            if terminated:
                current_game = False

            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            all_states.append(state)