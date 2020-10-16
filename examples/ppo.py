import os
import gym
import torch
from torch import nn
import kino_envs
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import VectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


class Conv(nn.Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Conv2d(1,1,kernel_size=(3,3), padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(1,1,kernel_size=(3,3), padding=1),
        ]
        self.model = nn.Sequential(*self.model)
        rand_sample = torch.rand((1,)+input_size)
        length = list(self.model(rand_sample).flatten().size())[0]
        print(length)
        self.linear = nn.Linear(length, output_size)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        logits = self.model(s)
        logits = torch.flatten(logits, 1)
        logits = self.linear(logits)
        return logits

class Actor(nn.Module):
    def __init__(self, layer, state_shape, img_shape, img_hidden_size, action_shape,
                 action_range, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """

        super().__init__()
        self.device = device
        self.conv = Conv(img_shape, img_hidden_size, device)
        
        if layer is None:
            layer = [512, 512, 128]
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, **kwargs):
        state = s.state
        local_map = s.local_map
        achieved_goal = s.achieved_goal
        desired_goal = s.desired_goal
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        if not isinstance(local_map, torch.Tensor):
            local_map = torch.tensor(local_map, device=self.device, dtype=torch.float)
        if not isinstance(achieved_goal, torch.Tensor):
            achieved_goal = torch.tensor(achieved_goal, device=self.device, dtype=torch.float)
        if not isinstance(desired_goal, torch.Tensor):
            desired_goal = torch.tensor(desired_goal, device=self.device, dtype=torch.float)

        hidden = self.conv(local_map)
        state[:,:2] = state[:,:2] - desired_goal[:,:2]
        # state = torch.cat((state,hidden), dim=1)
        logits = self.model(state)
        logits = torch.tanh(logits) * self.high
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, layer, state_shape, img_shape, img_hidden_size, action_shape,
                 action_range, device='cpu'):
        super().__init__()
        if layer is None:
            layer = [512, 512, 128]
        self.device = device
        self.conv = Conv(img_shape, img_hidden_size, device)
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(layer[-1], np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

    def forward(self, s, **kwargs):
        state = s.state
        local_map = s.local_map
        achieved_goal = s.achieved_goal
        desired_goal = s.desired_goal
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        if not isinstance(local_map, torch.Tensor):
            local_map = torch.tensor(local_map, device=self.device, dtype=torch.float)
        if not isinstance(achieved_goal, torch.Tensor):
            achieved_goal = torch.tensor(achieved_goal, device=self.device, dtype=torch.float)
        if not isinstance(desired_goal, torch.Tensor):
            desired_goal = torch.tensor(desired_goal, device=self.device, dtype=torch.float)

        hidden = self.conv(local_map)
        state[:,:2] = state[:,:2] - desired_goal[:,:2]
        # state = torch.cat((state,hidden), dim=1)
        logits = self.model(state)

        mu = torch.tanh(self.mu(logits)) * self.high
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class Critic(nn.Module):
    def __init__(self, layer, state_shape, img_shape, img_hidden_size, action_shape, device='cpu'):
        super().__init__()
        if layer is None:
            layer = [512, 512, 128]
        self.device = device
        self.conv = Conv(img_shape, img_hidden_size, device)
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        state = s.state
        local_map = s.local_map
        achieved_goal = s.achieved_goal
        desired_goal = s.desired_goal
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        if not isinstance(local_map, torch.Tensor):
            local_map = torch.tensor(local_map, device=self.device, dtype=torch.float)
        if not isinstance(achieved_goal, torch.Tensor):
            achieved_goal = torch.tensor(achieved_goal, device=self.device, dtype=torch.float)
        if not isinstance(desired_goal, torch.Tensor):
            desired_goal = torch.tensor(desired_goal, device=self.device, dtype=torch.float)

        hidden = self.conv(local_map)
        state[:,:2] = state[:,:2] - desired_goal[:,:2]
        # state = torch.cat((state,hidden), dim=1)

        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            s = torch.cat([state, a], dim=1)
        logits = self.model(s)
        return logits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='DifferentialDriveObs-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = gym.make(args.task)
    args.state_shape = env.observation_space['state'].shape or env.observation_space.n
    args.goal_shape = env.observation_space['desired_goal'].shape
    args.image_shape = env.observation_space['local_map'].shape
    args.action_shape = env.action_space.shape or env.action_space.n
    args.action_range = [env.action_space.low[0], env.action_space.high[0]]
    # train_envs = gym.make(args.task)
    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    args.img_hidden_size = 64
    conv = Conv(input_size=args.image_shape, output_size=args.img_hidden_size, device=args.device).to(args.device)
    actor = ActorProb(layer = [512, 128, 128], state_shape = args.state_shape, 
    img_shape=args.image_shape, 
    img_hidden_size=args.img_hidden_size, 
    action_shape = args.action_shape,
    action_range = args.action_range,
    device = args.device
    ).to(args.device)
    # actor.conv = conv

    critic = Critic(
        layer = [512, 128, 128],
        state_shape = args.state_shape,
        img_shape = args.image_shape,
        img_hidden_size = args.img_hidden_size,
        action_shape = args.action_shape,
        device = args.device
    ).to(args.device)


    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = DiagGaussian
    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        # dual_clip=args.dual_clip,
        # dual clip cause monotonically increasing log_std :)
        value_clip=args.value_clip,
        # action_range=[env.action_space.low[0], env.action_space.high[0]],)
        # if clip the action, ppo would not converge :)
        gae_lambda=args.gae_lambda)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=None, save_fn=save_fn,
        writer=writer)

    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_ppo()
