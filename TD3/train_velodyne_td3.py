import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv
from tqdm import tqdm

LR = 1e-4


def evaluate(network, epoch, epsilon, eval_episodes=10):
    """
    评估当前策略的性能，计算平均奖励和碰撞率
    """
    avg_reward = 0.0
    col = 0  # 碰撞计数
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()  # 复位环境
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))  # 获取当前策略的动作
            a_in = [action[0], action[1]]
            state, reward, done, _ = env.step(a_in)  # 执行动作并获取反馈
            avg_reward += reward
            count += 1
            if reward < -90:  # 发生碰撞
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print("Average Reward over %i Evaluation Episodes, Epoch:%i, avg_reward:%f, avg_col:%f, epsilon:%f" % (eval_episodes, epoch, avg_reward, avg_col, epsilon))
    print("..............................................")
    return avg_reward


class Actor(nn.Module):
    """
    生成动作的 Actor 网络。
    """

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()  # 输出动作的归一化

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """
    评估动作价值的 Critic 网络，采用双 Q 网络结构。
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 第一组 Q 网络
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        # 第二组 Q 网络
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # 计算 Q1 值
        s1 = F.relu(self.layer_1(s))
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        # 计算 Q2 值
        s2 = F.relu(self.layer_4(s))
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0  # 训练步数计数

    def get_action(self, state):
        """
        获取动作。
        """
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """
        训练 Actor 和 Critic 网络。
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0

        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # 计算目标动作
            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # 计算目标 Q 值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # 计算当前 Q 值
            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # 更新 Critic 网络
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # 更新 Actor 网络
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # 软更新目标网络
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        """
        加载模型参数。
        """
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))


# 设定设备：如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定训练超参数
seed = 500  # 随机种子，确保实验可复现
eval_freq = 5e3  # 评估间隔步数，每 5000 步进行一次评估
max_ep = 500  # 每个 episode 的最大步数
eval_ep = 10  # 评估时运行 10 个 episode
max_timesteps = 5e6  # 训练最大步数（500 万步）
expl_noise = 1  # 初始探索噪声（在 [expl_min, 1] 范围内）
expl_decay_steps = 500000  # 探索噪声衰减步数
expl_min = 0.1  # 探索噪声的最小值
batch_size = 40  # 训练时的小批量大小
discount = 0.995  # 折扣因子 γ（用于计算折扣奖励）
tau = 0.005  # 目标网络软更新系数
policy_noise = 0.2  # 目标动作上的噪声
noise_clip = 0.5  # 目标动作噪声的截断范围
policy_freq = 2  # 每 2 个训练步更新一次 Actor 网络
buffer_size = 1e6  # 经验回放缓冲区大小（最多存储 100 万条数据）
file_name = "TD3_velodyne"  # 保存策略的文件名
save_model = True  # 是否保存训练好的模型
load_model = True  # 是否加载已有模型
random_near_obstacle = False  # 是否在障碍物附近随机采取动作以增加探索性
greedy = False
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# 创建存储网络的文件夹
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# 创建训练环境
environment_dim = 20  # 观测环境的维度
robot_dim = 4  # 机器人状态维度
env = GazeboEnv("wheeltec_senior_akm.launch", environment_dim)  # 启动仿真环境
time.sleep(5)  # 等待 5 秒以确保环境稳定

# 设置随机种子，保证实验可复现
torch.manual_seed(seed)
np.random.seed(seed)

# 定义状态维度和动作维度
state_dim = environment_dim + robot_dim  # 总的状态维度 = 机器人 + 环境
action_dim = 2  # 动作维度（线速度和角速度）
max_action = 1  # 动作范围（假设动作在 [-1,1] 之间）

# 创建 TD3 训练网络
network = TD3(state_dim, action_dim, max_action)

# 创建经验回放缓冲区
replay_buffer = ReplayBuffer(buffer_size, seed)

# 如果启用了模型加载，则尝试加载已有模型
if load_model:
    try:
        network.load(file_name, r"../pytorch_models")
    except:
        print("无法加载模型，使用随机初始化的网络进行训练")

# 存储评估结果的列表
evaluations = []

# 初始化计数器
timestep = 0  # 训练步数
timesteps_since_eval = 0  # 自上次评估以来的步数
episode_num = 0  # 记录 episode 数量
done = True  # 是否完成一个 episode
epoch = 1  # 记录当前 epoch

count_rand_actions = 0  # 记录随机动作的持续时间
random_action = []  # 存储随机动作

# **训练循环**
# while timestep < max_timesteps:
for timestep in tqdm(range(int(max_timesteps)), unit="steps", desc="training process"):

    # **如果当前 episode 结束**
    if done:
        if timestep != 0:
            # 训练 TD3 网络
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        # **如果到达评估步数，进行评估**
        if timesteps_since_eval >= eval_freq:
            print("正在进行评估")
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate(network=network, epoch=epoch, epsilon=epsilon_start, eval_episodes=eval_ep))  # 评估计TD3
            network.save(file_name, directory=f"/home/ubuntu/Code/DRL-robot-navigation/TD3/pytorch_models")  # 保存模型
            np.save(f"/home/ubuntu/Code/DRL-robot-navigation/TD3/results", evaluations)  # 存储评估数据
            epoch += 1

        # **重置环境，开始新的一轮 episode**
        state = env.reset()
        done = False
        episode_reward = 0  # 当前 episode 的奖励
        episode_timesteps = 0  # 当前 episode 的步数
        episode_num += 1  # 记录 episode 数量

    # **更新探索噪声**
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # **获取策略网络的动作，并加上探索噪声**
    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

    # **在靠近障碍物时，强制随机采取动作，增加探索性**
    # if random_near_obstacle:
    #     if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1:
    #         count_rand_actions = np.random.randint(10, 20)  # 生成随机动作的持续步数
    #         random_action = np.random.uniform(-1, 1, 2)  # 生成随机动作
    #         random_action[0] = np.random.uniform(-0.5, 0)

    #     if count_rand_actions > 0:
    #         count_rand_actions -= 1
    #         action = random_action

    if greedy:
        if np.random.rand() < epsilon_start:
            action = np.random.uniform(-1, 1, 2)  # 随机动作（探索）
            action[0] = np.random.uniform(-1, 1)

        if timestep % 1000 == 0:
            if epsilon_start > epsilon_end:
                epsilon_start = epsilon_start * epsilon_decay

    # **调整动作范围**
    # 线速度调整到 [-1,1]，角速度保持 [-1,1]
    a_in = [action[0], action[1]]

    # **执行动作，获得新状态、奖励等信息**
    next_state, reward, done, target = env.step(a_in)

    # 记录是否终止（如果达到最大 episode 步数，则强制终止）
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward  # 累加当前 episode 的奖励

    # **存储经验到回放缓冲区**
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # **更新状态和计数器**
    state = next_state
    episode_timesteps += 1
    # timestep += 1
    timesteps_since_eval += 1

# **训练结束后，进行最终评估并保存模型**
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory=f"/home/ubuntu/Code/DRL-robot-navigation/TD3/pytorch_models")
np.save(f"/home/ubuntu/Code/DRL-robot-navigation/TD3/results", evaluations)
