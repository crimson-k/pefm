import torch
from torch import nn
from torch import distributions as torchd

from src.r2dreamer.rssm import Deter

import src.r2dreamer.distributions as dists
from src.r2dreamer.networks import LambdaLayer
from src.r2dreamer.tools import rpad, weight_init_

class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        self._obs_net = nn.Sequential() # posterior network
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential() # prior network
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state."""
        parameter = next(self.parameters())
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=parameter.dtype, device=parameter.device)
        stoch = torch.zeros(
            batch_size, self._stoch, self._discrete,
            dtype=parameter.dtype, device=parameter.device,
        )
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Roll out a shared recurrent state with prior and posterior branches."""
        post_stoch, deter = initial
        posts, deters, post_logits, priors, prior_logits = [], [], [], [], []
        for i in range(action.shape[1]):
            deter = self.recurrent(post_stoch, deter, action[:, i], reset[:, i])
            prior_stoch, prior_logit = self.prior(deter)
            post_stoch, post_logit = self.posterior(deter, embed[:, i])
            posts.append(post_stoch)
            deters.append(deter)
            post_logits.append(post_logit)
            priors.append(prior_stoch)
            prior_logits.append(prior_logit)
        return tuple(torch.stack(items, dim=1) for items in (
            posts, deters, post_logits, priors, prior_logits
        ))

    def recurrent(self, stoch, deter, transition_action, reset):
        """previous posterior state + transition action -> current deter."""
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        transition_action = torch.where(
            rpad(reset, transition_action.dim() - int(reset.dim())),
            torch.zeros_like(transition_action), transition_action,
        )
        return self._deter_net(stoch, deter, transition_action)

    def posterior(self, deter, embed):
        """current deter + real observation embedding -> posterior."""
        logit = self._obs_net(torch.cat([deter, embed], dim=-1))
        stoch = self.get_dist(logit).base_dist.probs.to(logit.dtype)
        return stoch, logit

    def prior(self, deter):
        """Predict the interaction state from history and action alone."""
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).base_dist.probs.to(logit.dtype)
        return stoch, logit

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
