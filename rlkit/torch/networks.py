"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
import pdb

from torch import distributions as pyd

def identity(x):
    return x

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            batch_attention=False,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            output_activation_half=False,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            use_dropout=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # when output is [mean, var], if output_activation_half is true ,just activate mean, not var
        self.output_activation_half = output_activation_half
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout
        self.fcs = []
        self.layer_norms = []
        self.dropouts = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
            
            if self.use_dropout:
                dropout_n = nn.Dropout(0.1)
                self.__setattr__("drop_out{}".format(i), dropout_n)
                self.dropouts.append(dropout_n)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
        self.batch_attention = batch_attention
        self.transition_attention = transformer.BatchTransitionAttention(
            hidden=100,
            input_size=input_size,
            output_size=input_size,
            n_layers=3,
            attn_heads=1,
            dropout=0.1
        ) if self.batch_attention else None

    def forward(self, input, return_preactivations=False):
        if self.batch_attention:
            input = self.transition_attention(input)
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.use_dropout and i < len(self.fcs) - 1:
                h = self.dropouts[i](h)
        preactivation = self.last_fc(h)
        half_output_size = int(self.output_size/2)
        if self.output_activation_half:
            output =  torch.cat([self.output_activation(preactivation[..., :half_output_size]), preactivation[..., half_output_size:]], dim=-1)
        else:
            output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(Mlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Proto(nn.Module):
    def __init__(self, proj_dim, pred_dim, T, num_protos, num_iters, topk, queue_size):
        super(Proto, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, proj_dim)
        )

        self.num_iters = 3
        self.T = T
        self.topk = topk
        self.num_protos = num_protos

        self.protos = nn.Linear(proj_dim, num_protos, bias=False)

        self.register_buffer('queue', torch.zeros(queue_size, proj_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.outputs = dict()
        self.apply(weight_init)


    def forward(self, s, t):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        s = self.predictor(s)
        s = F.normalize(s, dim=1, p=2)
        t = F.normalize(t, dim=1, p=2)

        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.T, dim=1)

        with torch.no_grad():
            scores_t = self.protos(t)
            q_t = self.sinkhorn(scores_t)

        loss = -(q_t * log_p_s).sum(dim=1).mean()
        return loss

    def only_match(self, s):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        s = self.predictor(s)
        s = F.normalize(s, dim=1, p=2)

        scores_s = self.protos(s)

        return scores_s

    def only_sinkhorn(self, s):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        q_s = self.sinkhorn(scores_s)

        return q_s


    def compute_reward(self, z):
        B = z.shape[0]
        Q = self.queue.shape[0]

        assert Q % self.num_protos == 0

        # normalize
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        z = F.normalize(z, dim=1, p=2)

        scores = self.protos(z).T
        p = F.softmax(scores, dim=1)
        idx = pyd.Categorical(p).sample()

        # enqueue
        ptr = int(self.queue_ptr[0])
        self.queue[ptr:ptr + self.num_protos] = z[idx]
        self.queue_ptr[0] = (ptr + self.num_protos) % Q

        # compute distances
        z_to_q = torch.norm(z[:, None, :] - self.queue[None, :, :], dim=2, p=2)
        d, _ = torch.topk(z_to_q, self.topk, dim=1, largest=False)
        reward = d[:, -1:]
        return reward


    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.T
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T


