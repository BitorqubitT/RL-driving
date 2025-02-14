"""
    Let AI play the game.
    Load pytorch model and correct params.
    Run the game.
"""
from game_env import Environment
import torch
from ppo_module import PPOagent
from utils.dataclass_helper import Args
from ppo_module import Memory

if __name__ == "__main__":

    args = Args(
    batch_size=[5000],
    gamma=[0.99],
    tau=[0.005],
    lr=1e-3,
    episodes=[6000],
    eps_start=[0.9],
    eps_end=[0.1],
    eps_decay=[1000],
    n_actions=[4],
    n_observations=[9],
    game_map="scuffed monza",
    architecture="2 layer 64",
    loss_function="",
    start_pos="static",
    minibatch_size= 1250,
    num_iterations= 4000,
    seed= 1,
    num_steps= [5000],
    num_envs= 1,
    update_epochs= [4],
    clip_coef= [0.2],
    clip_vloss= True,
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
    ppo_agent = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    
    file_name = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    ppo_agent.load(file_name)
    all_agents = [ppo_agent]

    env = Environment("player", args.game_map, args.start_pos, args.num_envs)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=ppodevice).unsqueeze(0)
    all_states = [state]
    current_game = True

    while current_game:

        all_actions = []
        
        for i, agent in enumerate(all_agents):
            action, _, _, _ = agent.get_action_and_value(all_states[i][0])
            all_actions.append(action)
        all_car_data = env.step(all_actions)

        all_states = []
        for car_data in all_car_data:
            
            observation, reward, terminated, truncated = car_data
            if terminated:
                current_game = False

            state = torch.tensor(observation, dtype=torch.float32, device=ppodevice).unsqueeze(0)
            all_states.append(state)