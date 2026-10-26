"""Clean-latent observations to RSSM prior/posterior and semantic predictions."""

import torch
from torch import nn
from torch.nn import functional as F

class Evaluator(nn.Module):
    def __init__(self, token_aggregator, rssm, target_dim=None):
        super().__init__()
        self.token_aggregator = token_aggregator
        self.rssm = rssm
        embed_size = token_aggregator.norm.normalized_shape[0]
        target_dim = embed_size if target_dim is None else target_dim
        self.posterior_reconstruction_head = nn.Linear(rssm.feat_size, target_dim)
        self.prior_prediction_head = nn.Linear(rssm.feat_size, target_dim)
        self.relation_head = nn.Sequential(
            nn.Linear(rssm.feat_size, 256), nn.SiLU(), nn.Linear(256, 3),
        )
        self.event_head = nn.Sequential(
            nn.Linear(rssm.feat_size, 256), nn.SiLU(), nn.Linear(256, 8),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(rssm.feat_size, 256), nn.SiLU(), nn.Linear(256, 5),
        )

    def _run_rssm(self, embed, actions, reset):
        parameter = next(self.rssm.parameters())
        embed = embed.to(device=parameter.device, dtype=parameter.dtype)
        actions = actions.to(device=parameter.device, dtype=parameter.dtype)
        reset = reset.to(device=parameter.device, dtype=torch.bool)
        initial = self.rssm.initial(embed.shape[0])
        post_stoch, deter, post_logits, prior_stoch, prior_logits = self.rssm.observe(
            embed, actions, initial, reset
        )
        posterior_features = self.rssm.get_feat(post_stoch, deter)
        prior_features = self.rssm.get_feat(prior_stoch, deter)
        posterior_reconstruction = self.posterior_reconstruction_head(posterior_features)
        prior_prediction = self.prior_prediction_head(prior_features)
        return {
            "embed": embed,
            "deter": deter,
            "posterior_stoch": post_stoch,
            "posterior_logits": post_logits,
            "prior_stoch": prior_stoch,
            "prior_logits": prior_logits,
            "posterior_reconstruction": posterior_reconstruction,
            "prior_prediction": prior_prediction,
            "posterior_relation_logits": self.relation_head(posterior_features),
            "prior_relation_logits": self.relation_head(prior_features),
            "posterior_event_logits": self.event_head(posterior_features),
            "prior_event_logits": self.event_head(prior_features),
            "posterior_motion_logits": self.motion_head(posterior_features),
            "prior_motion_logits": self.motion_head(prior_features),
        }

    def forward_embeddings(self, embed, actions, reset):
        """Run RSSM inference from precomputed per-step embeddings.

        ``embed`` is ``[B,T,E]`` and shares its time axis with actions/reset.
        """
        if embed.ndim != 3 or actions.ndim != 3 or reset.ndim != 2:
            raise ValueError(
                f"Expected embed [B,T,E], actions [B,T,A], reset [B,T]; "
                f"got {tuple(embed.shape)}, {tuple(actions.shape)}, {tuple(reset.shape)}"
            )
        if embed.shape[:2] != actions.shape[:2] or embed.shape[:2] != reset.shape:
            raise ValueError("embed, actions, and reset must have the same B,T dimensions")
        return self._run_rssm(embed, actions, reset)

    def forward_latents(self, latents, actions, reset):
        """Aggregate clean latents [B,48,T,H,W] into RSSM observations [B,T,512]."""
        if latents.ndim != 5 or latents.shape[1] != 48:
            raise ValueError(f"Expected clean latents [B,48,T,H,W], got {tuple(latents.shape)}")
        tokens = latents.permute(0, 2, 3, 4, 1).flatten(2, 3).contiguous()
        parameter = next(self.token_aggregator.parameters())
        embed = self.token_aggregator(tokens.to(parameter))
        output = self.forward_embeddings(embed, actions, reset)
        output["latent_tokens"] = tokens
        return output

    def forward(self, batch):
        return self.forward_latents(
            batch["clean_latents"], batch["action_delta"], batch["reset"],
        )

    @staticmethod
    def _masked_mean(value, valid):
        valid = valid.to(value)
        return (value * valid).sum() / valid.sum().clamp_min(1)

    def compute_loss(self, output, reset, free_nats=1.0, dyn_scale=1.0, rep_scale=0.1,
                     target_embed=None, interaction_target=None, interaction_mask=None,
                     interaction_weights=None, event_is_dual_arm=None,
                     interaction_is_multi_object=None):
        """Positive-pair losses; reset positions have no prediction target."""
        valid = ~reset
        target_embed = output["embed"] if target_embed is None else target_embed
        target_embed = target_embed.detach().to(output["posterior_reconstruction"])
        if target_embed.shape != output["posterior_reconstruction"].shape:
            raise ValueError(
                f"Target shape {tuple(target_embed.shape)} does not match prediction "
                f"{tuple(output['posterior_reconstruction'].shape)}"
            )

        reconstruction_error = (output["posterior_reconstruction"] - target_embed).square().mean(-1)
        prediction_error = (output["prior_prediction"] - target_embed).square().mean(-1)
        dyn_loss, rep_loss = self.rssm.kl_loss(
            output["posterior_logits"], output["prior_logits"], free_nats
        )

        losses = {
            "posterior_reconstruction": reconstruction_error.mean(),
            "prior_prediction": self._masked_mean(prediction_error, valid),
            "kl_dynamics": self._masked_mean(dyn_loss, valid),
            "kl_representation": self._masked_mean(rep_loss, valid),
        }
        losses["kl"] = dyn_scale * losses["kl_dynamics"] + rep_scale * losses["kl_representation"]
        losses["total"] = (
            losses["posterior_reconstruction"]
            + losses["prior_prediction"]
            + losses["kl"]
        )
        if interaction_target is not None:
            interaction_weights = {} if interaction_weights is None else interaction_weights
            losses.update(self.compute_interaction_loss(
                output, reset, interaction_target, interaction_mask, interaction_weights,
                event_is_dual_arm=event_is_dual_arm,
                interaction_is_multi_object=interaction_is_multi_object,
            ))
            losses["total"] = (
                losses["total"] + interaction_weights.get("interaction", 1.0) * losses["interaction"]
            )
        return losses

    def compute_interaction_loss(self, output, reset, target, mask=None, weights=None,
                                 event_is_dual_arm=None, interaction_is_multi_object=None):
        """将共享输出头与 [关系(3), 物理事件(8), 运动属性(5)] multi-hot GT 对齐。

        关系顺序：未接触、接触、抓持。事件按《最终方案书10.21》的表格顺序：
        空移，物体静止；空移，物体运动；机械臂静止，物体运动；双方静止；
        接触，物体运动；接触，物体静止；抓持，物体随动；抓持，物体静止。
        运动顺序：末端平移、末端旋转、
        夹爪张开、夹爪闭合、物体运动。
        mask 为 [B,T,16]，三类头均按有效维度计算 BCE。
        默认按单臂校验；event_is_dual_arm [B,T] 中为 True 的时间步允许多事件标签。
        interaction_is_multi_object [B,T] 标记按 OR 合并后的关系、事件 multi-hot。
        也支持未合并的 [B,T,K,16]，先 OR 合并，再计算一次损失。
        """
        if target.ndim == 4:
            if target.shape[:2] != reset.shape or target.shape[-1] != 16 or target.shape[2] < 1:
                raise ValueError(f"Expected interaction_target [B,T,K,16], got {tuple(target.shape)}")
            if mask is not None and mask.shape != target.shape:
                raise ValueError("interaction_mask must match interaction_target [B,T,K,16]")
            from pefm.data.dataloader import merge_object_labels
            interaction_is_multi_object = torch.full_like(reset, target.shape[2] > 1)
            object_mask = torch.ones_like(target, dtype=torch.bool) if mask is None else mask
            object_mask = object_mask & (~reset.to(target.device))[..., None, None]
            target, mask = merge_object_labels(target, object_mask)
        reference = output["posterior_relation_logits"]
        target = target.detach().to(device=reference.device, dtype=torch.float32)
        if target.shape != (*reset.shape, 16):
            raise ValueError(f"Expected interaction_target [B,T,16], got {tuple(target.shape)}")
        mask = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.to(
            device=reference.device, dtype=torch.bool,
        )
        if mask.shape != target.shape:
            raise ValueError("interaction_mask must have the same [B,T,16] shape as the target")
        mask = mask & (~reset.to(reference.device)).unsqueeze(-1)
        if (((target != 0) & (target != 1)) & mask).any():
            raise ValueError("Valid interaction labels must be binary; mask unknown labels")
        target = torch.where(mask, target, 0.0)
        if interaction_is_multi_object is None:
            interaction_is_multi_object = torch.zeros_like(reset, device=reference.device, dtype=torch.bool)
        else:
            interaction_is_multi_object = interaction_is_multi_object.to(device=reference.device, dtype=torch.bool)
            if interaction_is_multi_object.shape != reset.shape:
                raise ValueError("interaction_is_multi_object must have shape [B,T] matching reset")
        relation_mask = mask[..., :3]
        relation_positives = target[..., :3].sum(-1)
        invalid_relation = (relation_positives > 1) | (relation_mask.all(-1) & (relation_positives != 1))
        if (invalid_relation & ~interaction_is_multi_object).any():
            raise ValueError("Each valid single-object relation target must contain exactly one positive class")
        if event_is_dual_arm is None:
            event_is_dual_arm = torch.zeros_like(reset, device=reference.device, dtype=torch.bool)
        else:
            event_is_dual_arm = event_is_dual_arm.to(device=reference.device, dtype=torch.bool)
            if event_is_dual_arm.shape != reset.shape:
                raise ValueError("event_is_dual_arm must have shape [B,T] matching reset")
        event_mask = mask[..., 3:11]
        event_positives = target[..., 3:11].sum(-1)
        # 部分标签未知时只检查已知正例数量；全部已知时必须恰好有一个事件。
        invalid_event = (event_positives > 1) | (event_mask.all(-1) & (event_positives != 1))
        if (invalid_event & ~(event_is_dual_arm | interaction_is_multi_object)).any():
            raise ValueError(
                "Single-arm event targets must contain exactly one positive class when fully known, "
                "and at most one known positive class when partially masked"
            )
        weights = {} if weights is None else weights
        losses = {}
        for branch in ("posterior", "prior"):
            for name, selection in (("relation", slice(0, 3)), ("event", slice(3, 11)), ("motion", slice(11, 16))):
                error = F.binary_cross_entropy_with_logits(
                    output[f"{branch}_{name}_logits"].float(), target[..., selection], reduction="none",
                )
                losses[f"{branch}_{name}"] = self._masked_mean(error, mask[..., selection])
        for name in ("relation", "event", "motion"):
            losses[name] = losses[f"posterior_{name}"] + losses[f"prior_{name}"]
        losses["interaction"] = sum(
            weights.get(name, 1.0) * losses[name] for name in ("relation", "event", "motion")
        )
        return losses
