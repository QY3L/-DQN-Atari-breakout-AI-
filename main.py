import random
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from collections import deque
from atari_wrappers import wrap_deepmind, make_atari


# 训练或应用
TRAIN = False
# 应用CPU或GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建环境
env_name = 'Breakout'
env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
env = wrap_deepmind(env_raw, fire_reset=True, episode_life=True, clip_rewards=True)
env_test = wrap_deepmind(env_raw, fire_reset=True, episode_life=True, clip_rewards=False)


# 定义神经网络
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 以下参数论文给的

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8,8), stride=(4,), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2,), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1,), bias=False),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4),
        )

    def forward(self, x):
        x = self.features(x / 255.0)
        x = x.view(-1, 64*7*7)
        x = self.classifier(x)
        return x.float()


# 设计经验回放池
class ReplayMemory(object):
    def __init__(self, device):
        c, h, w = 5, 84, 84
        self.capacity = 500000
        self.device = device
        self.m_states = torch.zeros((self.capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((self.capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((self.capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((self.capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    # 存储
    def push(self, state, action, reward, done):

        self.m_states[self.position] = state
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = 1. if done else 0.
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    # 采样
    def sample(self,bs):
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4].to(self.device).float()
        bns = self.m_states[i, 1:].to(self.device).float()
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size


# 将4帧观测转化为合适尺寸的tensor
def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1, h, h).float()


# 用于训练
class Agent(object):

    def __init__(self):
        # 估计网络
        self.eval_net = DQN().to(device)
        # 目标网络
        self.target_net = DQN().to(device)
        # 保存参数
        torch.save(self.eval_net.state_dict(), 'model_weights.pth')
        # 拷贝参数至目标网络
        self.target_net.load_state_dict(torch.load('model_weights.pth'))
        # 目标网络仅用于估值
        self.target_net.eval()
        # 设定估计网络优化器
        self._optimizer = optim.Adam(self.eval_net.parameters(), lr=0.0000625, eps=1.5e-4)
        # 创建经验回放池
        self.memory = ReplayMemory(device)

        self._train_flag = False
        self._eps = 1.

    def select_action(self, ob):
        sample = random.random()
        # 线性退火
        if self._train_flag and self._eps > 0.1:
            self._eps -= 0.000001
        # 按策略选取动作
        if sample > self._eps:
            with torch.no_grad():
                a = self.eval_net(ob.to(device)).max(1)[1].view(1, 1)
        # 随机选取动作
        else:
            a = torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        return a

    # 训练神经网络
    def learn(self):
        self._train_flag = len(self.memory) > 500
        if not self._train_flag:
            return

        # 采样
        s, a, r, s_, if_done = self.memory.sample(32)

        # q_learning
        y_pre = self.eval_net(s.to(device)).gather(1, a)
        q_next = self.target_net(s_.to(device)).max(1)[0].detach().float()
        y_ture = ((q_next * 0.99) * (1. - if_done[:, 0]) + r[:, 0]).unsqueeze(1).float()

        # huber loss
        loss = F.smooth_l1_loss(y_pre, y_ture).float()

        # optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1).float()
        self._optimizer.step()


# 用于测试
class AgentTest(object):

    def __init__(self):
        self.policy_net = DQN()
        self.policy_net.load_state_dict(torch.load('model_weights1.pth', map_location='cpu'))
        self.policy_net.eval()

    def select_action(self, ob):
        sample = random.random()
        if sample > 0.01:
            a = self.policy_net(ob.to(device)).max(1)[1].view(1, 1)
        else:
            a = torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        return a


# 在测试环境中测试得分
def score_test():

    agent_test = AgentTest()
    done = True
    life = 5
    score = 0
    q = deque(maxlen=5)

    while True:
        if done:
            if life > 0:
                life -= 1
            else:
                break
            env_test.reset()
            for _ in range(5):
                n_frame, _, _, _ = env_test.step(0)
                n_frame = fp(n_frame)
                q.append(n_frame)

        # select and perform an action
        state = torch.cat(list(q))[1:].unsqueeze(0)
        action = agent_test.select_action(state)
        observation, reward, done, info = env_test.step(action)
        score += reward
        observation = fp(observation)
        # 5 frame as memory
        q.append(observation)

    return score

if __name__ == '__main__':
    if TRAIN:    # 训练
        agent = Agent()
        done = True
        # 记录连续5帧，前4帧为s，后4帧为s_
        q = deque(maxlen=5)
        # 记录进度并显示时间和进度条
        progressive = tqdm(range(10000000), total=10000000, ncols=50, leave=False, unit='b')

        for step in progressive:
            if done:
                env.reset()
                for _ in range(5):
                    ob, _, _, _ = env.step(0)
                    ob = fp(ob)
                    q.append(ob)

            # select and perform an action
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action = agent.select_action(state)
            observation, reward, done, info = env.step(action)
            observation = fp(observation)
            # 5 frame as memory
            q.append(observation)
            agent.memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done)

            if step % 4 == 0:
                agent.learn()

            if step % 10000 == 0:
                torch.save(agent.eval_net.state_dict(), 'model_weights.pth')
                agent.target_net.load_state_dict(torch.load('model_weights.pth'))

            # 每10000 步进行1次得分测试，记录在file中
            if step % 10000 == 0:
                score = score_test()
                f = open("file.txt", 'a')
                f.write("%f\n" % float(score))
                f.close()     #
    else:
        agent = AgentTest()
        done = True
        life = 5
        score = 0
        step = 0
        q = deque(maxlen=5)

        while True:
            env_test.render()
            step += 1
            if step > 12000:
                print('总分为：{}'.format(score))
                time.sleep(0.5)
                break
            if done:
                if life > 0:
                    life -= 1
                else:
                    print('总分为：{}'.format(score))
                    break
                env_test.reset()
                for _ in range(5):
                    n_frame, _, _, _ = env_test.step(0)
                    n_frame = fp(n_frame)
                    q.append(n_frame)

            state = torch.cat(list(q))[1:].unsqueeze(0)
            action = agent.select_action(state)
            observation, reward, done, info = env_test.step(action)
            score += reward
            time.sleep(0.01)
            observation = fp(observation)
            q.append(observation)
