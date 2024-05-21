import sys
sys.path.append('./env/')
from env import controller
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
import collections
import random
import time


DEFAULT_ID = 0
DEFAULT_BITRATE = 1
DEFAULT_SLEEP = 0
STEP = [5, 10 ,15]

# ----------------------------------- #
# 构建MasterPolicy Actor-Critic网络
# ----------------------------------- #

class MasterActor(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(MasterActor, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    def forward(self, x):
        x = x.float()
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b, n_actions]
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x

class MasterCritic(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(MasterCritic, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x

# class MasterActorCritic(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MasterActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_size),
#             nn.Softmax(dim=-1)
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, state):
#         action_probs = self.actor(state)
#         value = self.critic(state)
#         return action_probs, value

# class SubActorCritic(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(SubActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_size),
#             nn.Softmax(dim=-1)
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, state):
#         action_probs = self.actor(state)
#         value = self.critic(state)
#         return action_probs, value

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.m_actor = MasterActor(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.m_critic = MasterCritic(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.m_actor_optimizer = optim.Adam(self.m_actor.parameters(), lr = actor_lr)
        # 价值网络的优化器
        self.m_critic_optimizer = optim.Adam(self.m_critic.parameters(), lr = critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # 选择子策略
    def select_subpolicy(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # print("debug", state)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.m_actor(state)
        # 创建以probs为标准的概率分布
        action_list = Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action

    # 训练&更新参数
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1,1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.m_critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1-dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.m_critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.m_actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.m_actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.m_critic(states), td_target.detach()))

            # 梯度清0
            self.m_actor_optimizer.zero_grad()
            self.m_critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.m_actor_optimizer.step()
            self.m_critic_optimizer.step()


# # 定义PPO算法
# def ppo_update(policy, optimizer, states, actions, advantages, returns, clip_param=0.2, epochs=10):
#     for _ in range(epochs):
#         # 计算比率
#         action_probs, values = policy(states)
#         dist = Categorical(action_probs)
#         entropy = dist.entropy().mean()
#         new_values = values.squeeze(1)

#         # 计算advantages
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
#         # 计算policy loss
#         action_log_probs = dist.log_prob(actions)
#         ratio = torch.exp(action_log_probs - action_log_probs.detach())
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
#         policy_loss = -torch.min(surr1, surr2).mean()

#         # 计算value loss
#         value_loss = 0.5 * (new_values - returns).pow(2).mean()

#         # 总体损失
#         loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# ----------------------------------- #
# 构建SubPolicy Soft Actor-Critic网络
# ----------------------------------- #
# Value Net
class SubValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(SubValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# Soft Q Net
class SubSoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SubSoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)
        
    def forward(self, state, action):
        x = torch.cat([state,action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
    
        return x

# Policy Net
class SubPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, device, log_std_min = -20, log_std_max=2, edge=3e-3):
        super(SubPolicyNet, self).__init__()
        self.log_std_min = log_std_min 
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)
        
        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

        self.device = device
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
            
        return action
    
    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0,1)
        
        z = noise.sample()
        action = torch.tanh(mean + std*z.to(self.device))
        log_prob = normal.log_prob(mean + std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        
        return action, log_prob

class SubReplayBeffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(self.device), \
                torch.FloatTensor(action_list).to(self.device), \
                torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device), \
                torch.FloatTensor(next_state_list).to(self.device), \
                torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device):

        # self.env = env
        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.s_value_net = SubValueNet(state_dim).to(device)
        self.s_target_value_net = SubValueNet(state_dim).to(device)
        self.s_q1_net = SubSoftQNet(state_dim, action_dim).to(device)
        self.s_q2_net = SubSoftQNet(state_dim, action_dim).to(device)
        self.s_policy_net = SubPolicyNet(state_dim, action_dim, device).to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.s_target_value_net.parameters(), self.s_value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Initialize the optimizer
        self.s_value_optimizer = optim.Adam(self.s_value_net.parameters(), lr=value_lr)
        self.s_q1_optimizer = optim.Adam(self.s_q1_net.parameters(), lr=q_lr)
        self.s_q2_optimizer = optim.Adam(self.s_q2_net.parameters(), lr=q_lr)
        self.s_policy_optimizer = optim.Adam(self.s_policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = SubReplayBeffer(buffer_maxlen, device)

    def get_action(self, state):
        action = self.s_policy_net.action(state)

        return action

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.s_policy_net.evaluate(state)

        # V value loss
        value = self.s_value_net(state)
        new_q1_value = self.s_q1_net(state, new_action)
        new_q2_value = self.s_q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q loss
        q1_value = self.s_q1_net(state, action)
        q2_value = self.s_q2_net(state, action)
        target_value = self.s_target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # Update Policy
        self.s_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.s_policy_optimizer.step()

        # Update v
        self.s_value_optimizer.zero_grad()
        value_loss.backward()
        self.s_value_optimizer.step()

        # Update Soft q
        self.s_q1_optimizer.zero_grad()
        self.s_q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.s_q1_optimizer.step()
        self.s_q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.s_target_value_net.parameters(), self.s_value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

# # 定义MLSH模型
# class MLSHModel(nn.Module):
#     def __init__(self, input_size, num_subpolicies):
#         super(MLSHModel, self).__init__()
#         self.master_policy = MasterActorCritic(input_size, num_subpolicies)
#         self.subpolicies = [SubActorCritic(input_size, 1) for _ in range(num_subpolicies)]
#         self.num_subpolicies = num_subpolicies

#     def select_subpolicy(self, state):
#         master_probs, _ = self.master_policy(state)
#         subpolicy_idx = Categorical(master_probs).sample().item()
#         return subpolicy_idx
    


def train():
    
    
    # input_size = 10  # 输入状态的大小
    num_subpolicies = 6  # 子策略的数量
    device = torch.device('cuda') if torch.cuda.is_available() \
                            else torch.device('cpu')
    
    # Master参数
    m_state_dim = 35
    m_action_dim = 18
    num_episodes = 100  # 总迭代次数
    ppo_gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-2  # 价值网络的学习率
    n_hiddens = 128  # 隐含层神经元个数
    m_return_list = []  # 保存每个回合的return

    master_policy = PPO(n_states=m_state_dim,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层数
                n_actions=m_action_dim,  # 动作数
                actor_lr=actor_lr,  # 策略网络学习率
                critic_lr=critic_lr,  # 价值网络学习率
                lmbda = 0.95,  # 优势函数的缩放因子
                epochs = 10,  # 一组序列训练的轮次
                eps = 0.2,  # PPO中截断范围的参数
                gamma=ppo_gamma,  # 折扣因子
                device = device
                )
    
    # Sub参数
    s_state_dim = 15
    s_action_dim = 6
    tau = 0.01
    sac_gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 50000
    s_return_list = []  # 保存每个回合的return

    Episode = 100
    batch_size = 128

    sub_policies = [SAC(state_dim=s_state_dim, 
                        action_dim=s_action_dim,
                        gamma=sac_gamma, 
                        tau=tau, 
                        buffer_maxlen=buffer_maxlen, 
                        value_lr=value_lr, 
                        q_lr=q_lr, 
                        policy_lr=policy_lr,
                        device=device) 
                        for _ in range(num_subpolicies)]

    # mlsh_model = MLSHModel(input_size, num_subpolicies)
    # optimizer = optim.Adam(mlsh_model.parameters(), lr=1e-3)

    # 迭代训练
    max_epochs = 1
    T_warmup = 100  # warmup阶段迭代次数
    T_joint = 300  # 允许主策略和子策略共同更新的迭代次数

    for epoch in range(max_epochs): # 每个epoch选择一个session
        session_id = 166
        net_trace_id = 1
        net_quality = 'mixed'
        

        for iteration in range(T_warmup + T_joint):
            env = controller.Environment(session_id, net_trace_id, net_quality)
            # Initial the first step
            re_download_video_id = DEFAULT_ID
            bit_rate = DEFAULT_BITRATE
            sleep_time = DEFAULT_SLEEP
            m_state, s_state, s_reward, done = env.take_action(re_download_video_id, bit_rate, sleep_time)
            # m_state, s_state = env.reset()
            # done = False
            m_iter_return = 0  # 累计每回合的reward
            s_iter_return = 0
            s_action_range = [0, 5]
            # 构造数据集，保存每个回合的状态数据
            m_transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
            # s_states, s_actions, s_rewards, s_values, dones = [], [], [], [], []
            # m_rewards = []
            # s_reward = -1
            iteration_max_step = 5000
            step = 0
            while True:
                iteration_max_step -= 1
                if step == 0:
                    # 保存每个时刻的状态\动作\...
                    
                    # 获取状态
                    flattened_m_state = [item for sublist in m_state for item in sublist]
                    # print("m_state", flattened_m_state)
                    m_state_arr = np.array(flattened_m_state)
                    # print(m_state_arr)
                    # m_state_tensor = torch.tensor(flattened_m_state, dtype=torch.float32)
                    m_transition_dict['states'].append(m_state_arr)
                    
                    # m_state = torch.FloatTensor(m_state).unsqueeze(0)
                    # 选择子策略
                    start_time = time.time()
                    m_action = master_policy.select_subpolicy(m_state_arr)
                    # print("m_action", m_action)
                    m_transition_dict['actions'].append(m_action)
                    subpolicy_idx = m_action // len(STEP)
                    # print("subpolicy_idx", subpolicy_idx)
                    step = STEP[m_action % len(STEP)]
                    # print("step", step)

                # 选择动作
                flattened_s_state = [item for sublist in s_state for item in sublist]
                # print("s_state:", flattened_s_state)
                s_state_tensor = torch.tensor(flattened_s_state, dtype=torch.float32)
                s_action = sub_policies[subpolicy_idx].get_action(s_state_tensor)
                # action output range[-1,1],expand to allowable range
                # s_action_in_arr =  s_action * (s_action_range[1] - s_action_range[0]) / 2.0 +  (s_action_range[1] + s_action_range[0]) / 2.0
                # action_probs, s_value = sub_policy.subpolicies[subpolicy_idx](s_state)
                # dist = Categorical(s_action_in)
                # s_action = dist.sample().item()
                # s_action_in = round(s_action_in_arr[0])
                s_action_in = np.argmax(s_action)
                # print("s_action_in: ", s_action_in)
                if s_action_in == 5:
                    sleep_time = 0.5
                    re_download_video_id = 0
                else:
                    # print("s_action_in: ", s_action_in)
                    sleep_time = 0
                    re_download_video_id = s_action_in
                bit_rate = 1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                end_time = time.time()
                # 计算时间差
                elapsed_time = end_time - start_time
                
                # 打印运行时间
                print(f"代码运行时间: {elapsed_time} 秒")
                # 执行动作
                next_m_state, next_s_state, s_reward, done = env.take_action(re_download_video_id, bit_rate, sleep_time)

                # if end_of_video: # 子策略的reward是视频粒度的
                #     s_reward = Calsreward()

                done_mask = 0.0 if done else 1.0
                s_state_arr = np.array(flattened_s_state)
                flattened_next_s_state = [item for sublist in next_s_state for item in sublist]
                next_s_state_arr = np.array(flattened_next_s_state)
                sub_policies[subpolicy_idx].buffer.push((s_state_arr, s_action, s_reward, next_s_state_arr, done_mask))
                # 存储数据
                # s_states.append(s_state)
                # s_actions.append(s_action)
                # s_rewards.append(s_reward)
                # s_values.append(s_value)
                # dones.append(done)
                step -= 1

                # 累计回合奖励
                
                s_iter_return += s_reward
                s_iter_return = round(s_iter_return, 6)
                # print("s_iter_return", s_iter_return)
                # if s_iter_return <= -500:
                if iteration_max_step <= 0 and s_iter_return <= -500:
                    m_reward = s_reward # !!!!!!!!!!!!!!!!!!!!!
                    m_transition_dict['rewards'].append(m_reward)
                    flattened_next_m_state = [item for sublist in next_m_state for item in sublist]
                    m_transition_dict['next_states'].append(np.array(flattened_next_m_state))
                    m_transition_dict['dones'].append(done)
                    m_iter_return += m_reward
                    break

                if step == 0:
                    m_reward = s_reward # !!!!!!!!!!!!!!!!!!!!!
                    m_transition_dict['rewards'].append(m_reward)
                    flattened_next_m_state = [item for sublist in next_m_state for item in sublist]
                    m_transition_dict['next_states'].append(np.array(flattened_next_m_state))
                    m_transition_dict['dones'].append(done)
                    m_iter_return += m_reward



                if done:
                    break

                if iteration >= T_warmup:  # 在joint阶段才更新子策略
                    if sub_policies[subpolicy_idx].buffer.buffer_len() > 500:
                        sub_policies[subpolicy_idx].update(batch_size)

                m_state = next_m_state
                s_state = next_s_state

            # # 计算advantages和returns
            # next_value = mlsh_model.master_policy(torch.FloatTensor(next_state).unsqueeze(0))[1].item()
            # advantages, returns = compute_advantages_returns(rewards, values, dones, next_value)

            # # 转换为PyTorch张量
            # states = torch.cat(states)
            # actions = torch.LongTensor(actions)
            # advantages = torch.FloatTensor(advantages)
            # returns = torch.FloatTensor(returns)

            # 保存每个回合的return
            m_return_list.append(m_iter_return)
            s_return_list.append(s_iter_return)
            # 所有阶段都更新主策略（每回合结束后更新）
            master_policy.learn(m_transition_dict)

            # 打印回合信息
            print(f'iter:{iteration}, m_return:{np.mean(m_return_list[-10:])}, \
                    s_return:{s_iter_return}, buffer_capacity:{sub_policies[subpolicy_idx].buffer.buffer_len()}')  # buffer打印要调整！！！！！！！！！！！！！！



            # if iteration < T_warmup:
            #     # 仅在warmup阶段更新主策略
            #     optimizer.zero_grad()
            #     master_action_probs, _ = mlsh_model.master_policy(states)
            #     loss = -torch.log(master_action_probs[:, subpolicy_idx]).mean()
            #     loss.backward()
            #     optimizer.step()
            # else:
            #     # 在joint阶段更新主策略和子策略
            #     subpolicy_optimizer = optim.Adam(mlsh_model.subpolicies[subpolicy_idx].parameters(), lr=1e-3)
            #     ppo_update(mlsh_model.subpolicies[subpolicy_idx], subpolicy_optimizer, states, actions, advantages, returns)
            #     ppo_update(mlsh_model.master_policy, optimizer, states, actions, advantages, returns)


        # -------------------------------------- #
        # 绘图
        # -------------------------------------- #

        # plt.plot(m_return_list)
        plt.plot(s_return_list)
        plt.ylabel('Return')
        plt.xlabel("Episode")
        plt.grid(True)
        plt.title('return')
        plt.show()

if __name__ == '__main__':
    train()