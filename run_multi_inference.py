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
    start_pos="random",
    minibatch_size= 1250,
    num_iterations= 4000,
    seed= 1,
    num_steps= [5000],
    num_envs= 13,
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
    ppo_agent_2 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_3 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_4 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_5 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_6 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_7 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_8 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_9 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_10 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_11 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)
    ppo_agent_12 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)    
    ppo_agent_13 = PPOagent(memory, ppodevice, 9, 4, args.gamma[0], args.gae_lambda, args.batch_size[0], args.minibatch_size, args.update_epochs[0], args.clip_coef[0], args.lr).to(ppodevice)

    file_name1 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name2 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_610_321.88.pth"
    file_name3 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_627_302.46.pth"
    file_name4 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_630_301.43.pth"
    file_name5 = "saved_models/ppo/scuffed_monza_random/ppo_scuffed monza_random_613_412.32.pth"
    file_name6 = "saved_models/ppo/scuffed_monza_random/ppo_scuffed monza_random_617_408.42.pth"
    file_name7 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name8 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name9 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name10 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name11 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name12 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"
    file_name13 = "saved_models/ppo/scuffed monza static/ppo_scuffed monza_static_636_299.25.pth"

    ppo_agent.load(file_name1)
    ppo_agent_2.load(file_name2)
    ppo_agent_3.load(file_name3)
    ppo_agent_4.load(file_name4)
    ppo_agent_5.load(file_name5)
    ppo_agent_6.load(file_name6)
    ppo_agent_7.load(file_name7)
    ppo_agent_8.load(file_name8)
    ppo_agent_9.load(file_name9)
    ppo_agent_10.load(file_name10)
    ppo_agent_11.load(file_name11)
    ppo_agent_12.load(file_name12)
    ppo_agent_13.load(file_name13)
    
    all_agents = [ppo_agent, ppo_agent_2, ppo_agent_3, ppo_agent_4, ppo_agent_5, ppo_agent_6, ppo_agent_7, ppo_agent_8, ppo_agent_9, ppo_agent_10, ppo_agent_11, ppo_agent_12, ppo_agent_13]

    env = Environment("player", args.game_map, args.start_pos, args.num_envs)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=ppodevice).unsqueeze(0)
    all_states = [[state], [state], [state], [state], [state], [state], [state], [state], [state], [state], [state], [state], [state]] 

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

            state = torch.tensor(observation, dtype=torch.float32, device=ppodevice).unsqueeze(0)
            all_states.append(state)