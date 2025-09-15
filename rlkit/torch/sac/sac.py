import os
from turtle import position
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils
import pdb

class CSROSoftActorCritic(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            wandb_project_name=None,
            wandb_run_name=None,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            wandb_project_name=wandb_project_name,
            wandb_run_name=wandb_run_name,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']
        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.club_criterion                 = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf, self.c, self.encoder_sa, self.encoder_rs, self.generator, self.discriminator, self.club_model, self.proto = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.club_model_optimizer           = optimizer_class(self.club_model.parameters(), lr=self.context_lr)

        self.encoder_optimizer = optimizer_class(
            list(self.encoder_sa.parameters()) + list(self.encoder_rs.parameters()), lr=self.context_lr)
        self.generator_optimizer = optimizer_class(self.generator.parameters(), lr=0.0002)
        self.discriminator_optimizer = optimizer_class(self.discriminator.parameters(), lr=0.0002)

        self.n_update_gan = 5
        self.generator_dim = 20
        self.mi_mir_loss_weight = 5.0
        self.relabel_data_ratio = 0.9

        # Protype
        self.proto_optimizer = optimizer_class(self.proto.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.encoder_sa, self.encoder_rs, self.generator, self.discriminator, self.club_model, self.proto]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked


    def sample_context(self, indices, b_size=None):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        if b_size is None:
            batches = [ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
                idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(idx, batch_size=b_size, sequence=self.recurrent)) for
                       idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        return context

    def sample_sac_with_next(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = []
        next_batches = []
        for idx in indices:
            batch, next_batch = self.replay_buffer.random_batch_with_next(idx, batch_size=self.batch_size)
            batches.append(ptu.np_to_pytorch_batch(batch))
            next_batches.append(ptu.np_to_pytorch_batch(next_batch))

        unpacked = [self.unpack_batch_sac(batch, true_sparse_reward=self.is_true_sparse_rewards) for batch in batches]
        next_unpacked = [self.unpack_batch_sac(batch, true_sparse_reward=self.is_true_sparse_rewards) for batch in
                         next_batches]

        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        next_unpacked = [[x[i] for x in next_unpacked] for i in range(len(next_unpacked[0]))]
        next_unpacked = [torch.cat(x, dim=0) for x in next_unpacked]

        return unpacked, next_unpacked


    def sample_context_with_next(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        batches = []
        next_batches = []

        for idx in indices:
            batch, next_batch = self.replay_buffer.random_batch_with_next(idx, batch_size=self.embedding_batch_size,
                                                                              sequence=self.recurrent)
            batches.append(ptu.np_to_pytorch_batch(batch))
            next_batches.append(ptu.np_to_pytorch_batch(next_batch))

        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        next_context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in next_batches]
        # group like elements together
        next_context = [[x[i] for x in next_context] for i in range(len(next_context[0]))]
        next_context = [torch.cat(x, dim=0) for x in
                        next_context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            next_context = torch.cat(next_context[:-1], dim=2)
        else:
            next_context = torch.cat(next_context[:-2], dim=2)

        return context, next_context


    def generate_relabel_output(self, obs, indices):

        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        context_batch = self.sample_context(indices)
        context = context_batch[:, 0 * mb_size: 0 * mb_size + mb_size, :]

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_dim = 1

        with torch.no_grad():
            policy_outputs, task_z, _ = self.agent(obs, context, task_indices=indices)
            new_actions = policy_outputs[0]

            if len(obs.shape) == 2:
                b = obs.shape[0] // len(indices)
                eps = torch.randn((obs.shape[0], self.generator_dim), device=ptu.device)
            elif len(obs.shape) == 3:
                b = obs.shape[1]
                eps = torch.randn((obs.shape[0] * obs.shape[1], self.generator_dim), device=ptu.device)
                obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)

            fake_output = self.generator(len(indices), b, obs, new_actions, task_z, eps)
            if not self.use_next_obs_in_context:
                if len(obs.shape) == 2:
                    fake_output = fake_output[:, :1]
                elif len(obs.shape) == 3:
                    fake_output = fake_output[:, :, :1]

        relabel_context = torch.cat([obs, new_actions, fake_output], dim=-1).reshape(len(indices), b, -1).cuda()

        return relabel_context

    def get_relabel_context(self, relabel_num, indices, original_context_batch):

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_dim = 1

        total_num = relabel_num * 4

        context_relable_batch = self.sample_context(indices, total_num)
        context_relable_obs = context_relable_batch[:, :, :obs_dim]
        context_relable_obs = context_relable_obs.cuda()

        relabel_context = self.generate_relabel_output(context_relable_obs, indices)

        relabel_context = self.filter_relabel_context(relabel_context, relabel_num, original_context_batch)

        return relabel_context

    def get_relabel_context_without_filter(self, relabel_num, indices, original_context_batch):

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_dim = 1

        total_num = relabel_num * 1

        context_relable_batch = self.sample_context(indices, total_num)
        context_relable_obs = context_relable_batch[:, :, :obs_dim]
        context_relable_obs = context_relable_obs.cuda()

        relabel_context = self.generate_relabel_output(context_relable_obs, indices)

        return relabel_context

    def filter_relabel_context(self, relabel_context, relabel_num, original_context_batch):

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_dim = 1

        relabel_context_obs = relabel_context[:, :, :obs_dim]
        relabel_context_actions = relabel_context[:, :, obs_dim:obs_dim+action_dim]
        relabel_context_rewards = relabel_context[:, :, obs_dim+action_dim].unsqueeze(-1)
        relabel_context_next_obs = relabel_context[:, :, obs_dim+action_dim+1:]

        t, b, _ = relabel_context_obs.size()

        sa_repre = self.encoder_sa(t, b, relabel_context_obs, relabel_context_actions)
        rs_repre = self.encoder_rs(t, b, relabel_context_rewards, relabel_context_next_obs)

        if len(sa_repre.shape) == 2 and len(rs_repre.shape) == 2:
            logits = torch.einsum('iz,jz->ij', sa_repre, rs_repre)
        else:
            logits = torch.einsum('eiz,ejz->eij', sa_repre, rs_repre)


        sa_repre = torch.linalg.norm(sa_repre, dim=-1, keepdim=True)
        rs_repre = torch.linalg.norm(rs_repre, dim=-1, keepdim=True)

        diagonal_elements = torch.diagonal(logits, dim1=-2, dim2=-1).reshape(sa_repre.shape[0], sa_repre.shape[1], 1)
        src_info = diagonal_elements / (sa_repre * rs_repre)
        src_info = src_info.squeeze(-1)

        sorted_indices = torch.argsort(src_info, dim=-1)
        top_indices = sorted_indices[:, -relabel_num:]
        top_indices = top_indices.unsqueeze(-1).expand(-1, -1, relabel_context.shape[-1])
        true_relabel_context = torch.gather(relabel_context, 1, top_indices)


        return true_relabel_context


    def _relabel(self, context_batch, indices):

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_dim = 1

        task_num = context_batch.shape[0]
        data_num = context_batch.shape[1]
        relabel_num = int(data_num * self.relabel_data_ratio)

        original_context_batch = context_batch[:, :-relabel_num, :]
        relabel_context_batch = context_batch[:, -relabel_num:, :]

        relabel_context = self.get_relabel_context_without_filter(relabel_num, indices, original_context_batch)

        new_context_batch = torch.cat([original_context_batch, relabel_context], dim=1)

        return new_context_batch

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        context_batch, next_context_batch = self.sample_context_with_next(indices)

        context_batch = self._relabel(context_batch, indices)
        next_context_batch = self._relabel(next_context_batch, indices)

        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            next_context = next_context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars, wandb_stat = self._take_step(indices, context, next_context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars, wandb_stat

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        
    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * b # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    def _discriminator_loss(self, t, b, real_rewards, real_next_obs, fake_rewards, fake_next_obs, obs, actions, task_z):

        criterion1 = nn.BCELoss(reduction='mean')
        criterion2 = nn.BCELoss(reduction='mean')

        out_real = self.discriminator(t, b, obs, actions, real_rewards, real_next_obs, task_z)
        out_fake = self.discriminator(t, b, obs, actions, fake_rewards, fake_next_obs, task_z)

        target_real = torch.ones((t * b, 1), device=ptu.device)
        target_fake = torch.zeros((t * b, 1), device=ptu.device)

        d_loss_real = criterion1(out_real, target_real)
        d_loss_fake = criterion2(out_fake, target_fake)

        return d_loss_real, d_loss_fake

    def _generator_loss(self, t, b, fake_rewards, fake_next_obs, obs, actions, task_z):

        criterion = nn.BCELoss(reduction='mean')
        out = self.discriminator(t, b, obs, actions, fake_rewards, fake_next_obs, task_z)
        targets = torch.ones((t * b, 1), device=ptu.device)
        loss = criterion(out, targets)

        return loss

    def _IGDF_loss(self, t, b, real_rewards, real_next_obs, fake_rewards, fake_next_obs, obs, actions, task_z):

        real_rewards = real_rewards.reshape(t, b, -1)
        real_next_obs = real_next_obs.reshape(t, b, -1)
        fake_rewards = fake_rewards.reshape(t, b, -1)
        fake_next_obs = fake_next_obs.reshape(t, b, -1)

        obs = obs.reshape(t, b, -1)
        actions = actions.reshape(t, b, -1)
        task_z = task_z.reshape(t, b, -1)

        obs = obs.unsqueeze(-2)
        actions = actions.unsqueeze(-2)
        task_z = task_z.unsqueeze(-2)

        real_rewards = real_rewards.unsqueeze(-2)
        real_next_obs = real_next_obs.unsqueeze(-2)

        fake_rewards = fake_rewards[:, :-1, :]
        fake_next_obs = fake_next_obs[:, :-1, :]

        fake_rewards = fake_rewards.unsqueeze(1)
        fake_next_obs = fake_next_obs.unsqueeze(1)

        fake_rewards = fake_rewards.expand(-1, b, -1, -1)
        fake_next_obs = fake_next_obs.expand(-1, b, -1, -1)

        rr = torch.cat((real_rewards, fake_rewards), dim=-2)
        nss = torch.cat((real_next_obs, fake_next_obs), dim=-2)


        sa_repre = self.encoder_sa(t, b, obs, actions, task_z)
        rs_repre = self.encoder_rs(t, b, rr, nss)

        logits = torch.einsum('eliz,eljz->elij', sa_repre, rs_repre)
        logits = logits.squeeze(-2)

        matrix = torch.zeros((t, b, b), dtype=torch.float32,
                             device=ptu.device)

        matrix[:, :, 0] = 1
        info_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, matrix)
        info_loss = torch.mean(info_loss)

        return info_loss

    def InfoNCE_loss(self, context_embedding_1, context_embedding_2, temperature=0.1):


        context_embedding_1 = F.normalize(context_embedding_1, p=2, dim=1)  # (N, D)
        context_embedding_2 = F.normalize(context_embedding_2, p=2, dim=1)  # (N, D)

        logits = torch.matmul(context_embedding_1, context_embedding_2.T) / temperature
        labels = torch.arange(logits.shape[0]).long().to(logits.device)

        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss

    def make_contrastive_loss(self):
        total_indices = self.train_tasks
        total_context, total_next_context = self.sample_context_with_next(total_indices)

        task_z_protype = self.agent.encode_mean(total_context)
        clue_task_z_protype = self.club_model(total_context[..., :self.club_model.input_size])
        clue_task_z_protype = clue_task_z_protype[..., :self.latent_dim]
        clue_task_z_protype = torch.mean(clue_task_z_protype, dim=1).cuda()
        next_task_z_protype = self.agent.encode_mean(total_next_context)
        next_clue_task_z_protype = self.club_model(total_next_context[..., :self.club_model.input_size])
        next_clue_task_z_protype = next_clue_task_z_protype[..., :self.latent_dim]
        next_clue_task_z_protype = torch.mean(next_clue_task_z_protype, dim=1).cuda()

        task_z_protype_match = self.proto.only_match(task_z_protype)
        clue_task_z_protype_match = self.proto.only_match(clue_task_z_protype)
        next_task_z_protype_match = self.proto.only_match(next_task_z_protype)
        next_clue_task_z_protype_match = self.proto.only_match(next_clue_task_z_protype)

        task_z_protype_match_normalized = F.normalize(task_z_protype_match, p=2, dim=-1)
        cosine_similarity_matrix = torch.mm(task_z_protype_match_normalized, task_z_protype_match_normalized.t())

        task_num = cosine_similarity_matrix.size(0)
        cosine_similarity_matrix.fill_diagonal_(-float('inf'))

        topk = 8
        _, topk_indices = torch.topk(cosine_similarity_matrix, topk, dim=-1)

        loss = 0.
        for i in range(task_num):
            current_tasks = topk_indices[i, :]
            current_task_z_protype_match = task_z_protype_match[current_tasks, :]
            current_clue_task_z_protype_match = clue_task_z_protype_match[current_tasks, :]
            current_next_task_z_protype_match = next_task_z_protype_match[current_tasks, :]
            current_next_clue_task_z_protype_match = next_clue_task_z_protype_match[current_tasks, :]

            clue_task_z_loss = self.InfoNCE_loss(current_clue_task_z_protype_match, current_next_clue_task_z_protype_match).mean()
            next_task_z_loss = self.InfoNCE_loss(current_task_z_protype_match, current_next_task_z_protype_match).mean()

            loss = loss + clue_task_z_loss + next_task_z_loss

        loss = loss / task_num

        return loss


    def _take_step(self, indices, context, next_context):

        wandb_stat = {}


        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)
        wandb_stat["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)


        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()
        wandb_stat["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        wandb_stat["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()
        wandb_stat["target_v_values"] = torch.mean(target_v_values).item()

        for _ in range(self.n_update_gan):
            eps = torch.randn((t * b, self.generator_dim), device=ptu.device)
            fake_output = self.generator(t, b, obs, actions, task_z, eps)
            fake_rewards = fake_output[:, 0].unsqueeze(-1)
            fake_next_obs = fake_output[:, 1:]

            d_loss_real, d_loss_fake = self._discriminator_loss(t, b, rewards.reshape(t*b,-1), next_obs.reshape(t*b,-1),
                                                                fake_rewards.reshape(t*b,-1), fake_next_obs.reshape(t*b,-1),
                                                                obs.reshape(t*b,-1), actions.reshape(t*b,-1), task_z.reshape(t*b,-1))

            self.discriminator_optimizer.zero_grad()
            discriminator_loss = d_loss_real + d_loss_fake
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()

            IGDF_rewards = rewards.reshape(t, b, -1)
            IGDF_next_obs = next_obs.reshape(t, b, -1)
            IGDF_fake_rewards = fake_rewards.reshape(t, b, -1)
            IGDF_fake_next_obs = fake_next_obs.reshape(t, b, -1)

            IGDF_obs = obs.reshape(t, b, -1)
            IGDF_actions = actions.reshape(t, b, -1)
            IGDF_task_z = task_z.reshape(t, b, -1)

            info_loss = self._IGDF_loss(t, 32, IGDF_rewards[:, :32, :], IGDF_next_obs[:, :32, :],
                                        IGDF_fake_rewards[:, :32, :], IGDF_fake_next_obs[:, :32, :],
                                        IGDF_obs[:, :32, :], IGDF_actions[:, :32, :], IGDF_task_z[:, :32, :])
            info_loss = info_loss * 1.0
            info_loss.backward(retain_graph=True)
            self.encoder_optimizer.step()
            generator_loss = self._generator_loss(t, b, fake_rewards, fake_next_obs, obs, actions, task_z)
            generator_loss.backward(retain_graph=True)
            self.generator_optimizer.step()


        self.loss["generator_loss"] = generator_loss.item()
        wandb_stat["generator_loss"] = torch.mean(generator_loss).item()
        self.loss["info_loss"] = info_loss.item()
        wandb_stat["info_loss"] = torch.mean(info_loss).item()
        self.loss['discriminator_loss'] = discriminator_loss.item()
        wandb_stat["discriminator_loss"] = torch.mean(discriminator_loss).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        self.club_model_optimizer.zero_grad()
        self.proto_optimizer.zero_grad()


        if self.use_club:

            z_target = self.agent.encode_no_mean(context).detach()
            z_param = self.club_model(context[...,:self.club_model.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            club_model_loss = self.club_model_loss_weight * ((z_target - z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
            # club_model_loss.backward(retain_graph=True)
            self.loss["club_model_loss"] = club_model_loss.item()
            wandb_stat["club_model_loss"] = club_model_loss.item()

            z_target = self.agent.encode_no_mean(context)
            z_param = self.club_model(context[...,:self.club_model.input_size]).detach()
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            z_t, z_b, _ = z_mean.size()
            position = - ((z_target-z_mean)**2/z_var).mean()
            z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_target_repeat = z_target.repeat(1, z_b, 1)
            negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
            club_loss = self.club_loss_weight * (position - negative)
            # club_loss.backward(retain_graph=True)
            self.loss["club_loss"] = club_loss.item()
            wandb_stat["club_loss"] = club_loss.item()

            total_club_loss = club_model_loss + club_loss
            total_club_loss = total_club_loss * 1.0
            total_club_loss.backward(retain_graph=True)

        task_z_protype = self.agent.encode_mean(context)
        clue_task_z_protype = self.club_model(context[...,:self.club_model.input_size])
        clue_task_z_protype = clue_task_z_protype[..., :self.latent_dim]
        clue_task_z_protype = torch.mean(clue_task_z_protype, dim=1).cuda()


        next_task_z_protype = self.agent.encode_mean(next_context)
        next_clue_task_z_protype = self.club_model(next_context[..., :self.club_model.input_size])
        next_clue_task_z_protype = next_clue_task_z_protype[..., :self.latent_dim]
        next_clue_task_z_protype = torch.mean(next_clue_task_z_protype, dim=1).cuda()

        next_protype_loss = self.proto(task_z_protype, next_task_z_protype).mean()
        next_protype_loss = next_protype_loss * 1.0

        self.loss["next_protype_loss"] = next_protype_loss.item()
        wandb_stat["next_protype_loss"] = next_protype_loss.item()

        protype_loss = self.proto(clue_task_z_protype, next_clue_task_z_protype).mean()
        protype_loss = protype_loss * 1.0
        self.loss["clue_protype_loss"] = protype_loss.item()
        wandb_stat["clue_protype_loss"] = protype_loss.item()

        total_protype_loss = protype_loss * 1.0 + \
                             next_protype_loss * 1.0

        self.loss["total_protype_loss"] = total_protype_loss.item()
        wandb_stat["total_protype_loss"] = total_protype_loss.item()
        total_protype_loss.backward(retain_graph=True)


        # contrastive learning
        task_z_protype_match = self.proto.only_match(task_z_protype)
        clue_task_z_protype_match = self.proto.only_match(clue_task_z_protype)
        # protype_contrastive_loss = self.InfoNCE_loss(task_z_protype_match, clue_task_z_protype_match)

        next_task_z_protype_match = self.proto.only_match(next_task_z_protype)
        next_clue_task_z_protype_match = self.proto.only_match(clue_task_z_protype)
        next_protype_contrastive_loss = self.InfoNCE_loss(
            task_z_protype_match, next_task_z_protype_match
        )

        clue_protype_contrastive_loss = self.InfoNCE_loss(
            clue_task_z_protype_match, next_clue_task_z_protype_match
        )

        next_protype_contrastive_loss = self.InfoNCE_loss(
            task_z_protype_match, next_task_z_protype_match
        )

        total_contrastive_loss = clue_protype_contrastive_loss * 1.0 + \
                                 next_protype_contrastive_loss * 1.0
        total_contrastive_loss = total_contrastive_loss * 1.0
        total_contrastive_loss.backward(retain_graph=True)

        new_contrastive_loss = self.make_contrastive_loss()
        new_contrastive_loss = new_contrastive_loss * 1.0
        new_contrastive_loss.backward(retain_graph=True)


        if self.use_FOCAL_cl:
            z_loss = self.z_loss_weight * self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
            wandb_stat["z_loss"] = z_loss.item()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        wandb_stat["qf_loss"] = qf_loss.item()
        wandb_stat["q_target"] = torch.mean(q_target).item()
        wandb_stat["q1_pred"] = torch.mean(q1_pred).item()
        wandb_stat["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()
        self.club_model_optimizer.step()
        self.proto_optimizer.step()


        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        wandb_stat["vf_loss"] = vf_loss.item()
        wandb_stat["v_target"] = torch.mean(v_target).item()
        wandb_stat["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        wandb_stat["policy_loss"] = policy_loss.item()

        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        wandb_stat["a_loss"] = a_loss.item()

        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean


            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]

            if self.use_club:
                self.eval_statistics['Club model Loss'] = ptu.get_numpy(club_model_loss)
                self.eval_statistics['Club Loss'] = ptu.get_numpy(club_loss)
            if self.use_FOCAL_cl:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha',          ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate',   ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars), wandb_stat

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            club_model=self.club_model.state_dict(),
            c=self.c.state_dict(),
            )
        return snapshot