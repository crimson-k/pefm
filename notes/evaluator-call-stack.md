# PEFM Stage 1 Evaluator：BWM → V-JEPA → RSSM 调用栈

这份 note 对应 Stage 1 的 Day 3–6：先贯通正确配对数据的 observation/dynamics 路径。它在 R2Dreamer 中占据 `Dreamer.update() -> Dreamer._cal_grad()` 的位置，但不迁移 replay、actor、critic 或环境逻辑。原调用位置是 [`Dreamer.update()`](/data1/fangxuebin/pefm/src/r2dreamer/dreamer.py:310)，原 world-model 前三步是 encoder、posterior rollout 和 prior，见 [`Dreamer._cal_grad()`](/data1/fangxuebin/pefm/src/r2dreamer/dreamer.py:367)。

## 1. 当前最小边界

```text
BWM RoboTwinUnifiedDataset
  -> EvaluatorDataset / normal DataLoader
  -> RGB [B,V,3,81,H,W]
  -> VJEPAObservationAdapter
  -> visual_tokens [B,21,V*S,Dv]
  -> TokenAggregator
  -> real_embed [B,21,E]
  -> RSSM.observe(real_embed, grouped_eef)
  -> posterior stochastic states + deter
  -> RSSM.prior(deter)
  -> prior stochastic states
```

BWM 的 dataset 已经负责 metadata、视频 operator、EEF operator 和显式 frame indices，薄封装只读取其结果，不再次解析 RoboTwin。[`RoboTwinUnifiedDataset.__getitem__()`](/data1/fangxuebin/boundless-world-model/wan_video_action/data/wan_dataset.py:67) 是真实上游入口；[`EvaluatorDataset.__getitem__()`](/data1/fangxuebin/pefm/pefm/data/evaluator.py:29) 将其整理成 Evaluator batch。

当前输入是 BWM 已归一化的 14 维 EEF。具体使用 `eef_abs` 还是 `eef_delta` 仍由 BWM 的 `LoadCobotAction` 和数据列决定，wrapper 不擅自改变动作语义。

当前 prior 是标准 action-conditioned RSSM prior：EEF 与上一步 stochastic state 先产生 `deter`，prior 只读取 `deter`；真实视觉 embedding 只进入 posterior。[`RSSM.obs_step()`](/data1/fangxuebin/pefm/pefm/models/rssm_core.py:91) 与 [`RSSM.prior()`](/data1/fangxuebin/pefm/pefm/models/rssm_core.py:122) 保持这条因果边界。V-JEPA2-AC predictor 需要额外明确 proprio/history 协议，因此不塞进这次仅指定 encoder→RSSM 的最小实现。

## 2. 时间和张量对齐

81 帧使用同一张 `group_ids`：第 0 帧单独成组，剩余 80 帧每 4 帧一组，因此共有 21 组。[`_group_ids()`](/data1/fangxuebin/pefm/pefm/data/evaluator.py:7) 同时用于 RGB token 和 EEF，避免两边独立计算产生偏移。

```text
group 0      scene frame 0                 reset=True
group 1..2   recent history 8 frames       posterior burn-in
group 3..20  future 72 frames              future prior/posterior comparison
```

EEF 在每组内取均值，得到 `[B,21,14]`；RGB 不先压成图像均值，而是逐帧送入 V-JEPA，再对同一空间位置的四帧 token 求组均值，得到 `[B,21,V*S,Dv]`。多视角只合并到 token 维，不拼接图像高度。[`EvaluatorDataset`](/data1/fangxuebin/pefm/pefm/data/evaluator.py:38) 和 [`VJEPAObservationAdapter.forward()`](/data1/fangxuebin/pefm/pefm/models/vjepa_adapter.py:34) 分别实现两侧分组。

adapter 将 BWM 的 `[-1,1]` RGB 恢复到 `[0,1]`，保持宽高比 resize + center crop，再做 ImageNet normalization。每个原始帧复制两次形成 tubelet，这与官方 AC target encoder 的处理一致：[`forward_target()`](/data1/fangxuebin/vjepa2/app/vjepa_droid/train.py:408)。

`TokenAggregator` 用少量 learnable queries 对 view/spatial tokens 做 cross-attention，最终只输出固定宽度的 [`real_embed [B,T,E]`](/data1/fangxuebin/pefm/pefm/models/vjepa_adapter.py:65)。

## 3. 宏观 evaluator 伪代码

下面是建议的 `scripts/train_evaluator.py` 宏观位置。它是调用栈说明，不要求现在新增训练框架。

```python
def main(cfg):
    # BWM 仍是数据事实来源：metadata -> frame_indices -> video/action operators
    bwm_dataset = RoboTwinUnifiedDataset(
        base_path=cfg.data_root,
        metadata_path=cfg.train_manifest,
        data_file_keys=("video", "action"),
        main_data_operator=bwm_video_operator,
        special_operator_map={"action": bwm_eef_operator},
        temporal_template_sampling=True,
        temporal_num_frames=81,
        temporal_num_history_frames=9,
    )
    loader = build_evaluator_dataloader(bwm_dataset, cfg.batch_size)

    # Encoder/checkpoint 的构造继续使用 V-JEPA2 上游代码；adapter 只包输入输出协议。
    vjepa_encoder = load_vjepa_encoder(cfg.vjepa_checkpoint)
    evaluator = VJEPARSSMEvaluator(
        vjepa_adapter=VJEPAObservationAdapter(vjepa_encoder, cfg.crop_size),
        token_aggregator=TokenAggregator(cfg.token_dim, cfg.embed_dim),
        rssm=RSSM(cfg.rssm, embed_size=cfg.embed_dim, act_dim=14),
    ).to(cfg.device)

    for batch in loader:
        batch = move_to_device(batch, cfg.device)
        output = evaluator(batch)                 # 相当于 Dreamer._cal_grad() 前三步

        future = slice(3, 21)                     # 前 3 组是 scene/history burn-in
        dyn_loss, rep_loss = evaluator.rssm.kl_loss(
            output["posterior_logits"][:, future],
            output["prior_logits"][:, future],
            cfg.kl_free,
        )
        prediction_loss = feature_head_loss(
            output["prior_stoch"][:, future],
            output["real_embed"][:, future].detach(),
        )
        loss = dyn_loss.mean() + rep_loss.mean() + prediction_loss
        optimizer_step(loss)
```

`VJEPARSSMEvaluator.forward()` 是宏观模型入口：adapter → aggregator → initial → posterior rollout → prior，一共只保留数据流必须的五步，[`forward()`](/data1/fangxuebin/pefm/pefm/models/vjepa_rssm.py:13)。RSSM 内部再按时间调用 [`observe()`](/data1/fangxuebin/pefm/pefm/models/rssm_core.py:73)，每一步 posterior sample 会进入下一步 recurrent transition。

## 4. 已验证内容与剩余接口

最小验证覆盖：

- mock BWM dataset 正常 collate，`81 -> 21` 的 RGB/EEF 分组一致；
- dummy encoder 的完整 forward 得到 `real_embed`、posterior logits 和 prior logits；
- encoder 参数冻结时，`real_embed` 仍可向 RGB 输入反传梯度，便于后续冻结接入 BWM；
- 上游真实 `vit_tiny` encoder 可被 adapter 调用，并保持空间 token；
- 真实 BWM metadata/operator 抽样得到 `rgb [1,1,3,81,192,256]`、`eef [1,21,14]`、`group_ids [1,81]` 和 `reset [1,21]`。

当前未做训练，也没有可用 V-JEPA checkpoint，因此未验证真实 checkpoint 的 preprocessing 等价性和 embedding 数值。真实 task1 parquet 只有 `observation.state` 列，抽样时必须配置 `eef_abs`；若正式数据改用含 `action` 列的 manifest，才配置 `eef_delta`。另外，靠近 episode 起点的 BWM window 会把不足的 history clamp 到 frame 0；dataset 已把真实 `frame_indices` 原样返回，后续 alignment/admission 测试应显式统计这些重复历史帧。
