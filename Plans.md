# DINOSAM Plans.md

作成日: 2026-06-19

---

## Spec delta

- **path**: `docs/superpowers/specs/2026-06-19-dino-adapter-selectable-design.md`
- **change**: 以下の 7 点を product contract に反映する
- **why**: code review で検出されたバグ・設計不備を実装前に仕様化し、task contract の前提とするため

| # | 種別 | 変更点 |
|---|------|--------|
| 1 | **CRITICAL fix** | Section 4.3: DualAttnAdapter の identity 二重加算を修正。PAM_Module / CAM_Module 内部の `+ x` (residual) を削除し、DualAttnAdapter.forward() で 1 回だけ `+ x_2d` する方式に変更。または `(pam_out + cam_out) / 2` で正規化する方式も可（後者のほうが局所的変更で済む） |
| 2 | **HIGH fix** | Section 4.3: hw_shapes=None ガードを追加。None の場合は `ValueError("DualAttnAdapter requires hw_shapes")` を raise（2D reshape が必須のため、Mona のような fallback 不可） |
| 3 | **MEDIUM fix** | Section 4.3: factor パラメータを PAM_Module の reduction ratio に配線。query/key conv の出力チャネル数を `C // factor` とする（硬直値 `/8` からの変更） |
| 4 | **HIGH doc** | Section 4.3: PAM のメモリ使用量に関する注意書きを追加。64x64 features で 4096x4096 の attention map（float32 で約 67MB/sample）、batch_size=4 で約 268MB |
| 5 | **HIGH compat** | Section 5.1: checkpoint の後方互換性に関する注記を追加。mona1/mona2 → adapter1/adapter2 の key 名変更により旧 checkpoint の直接ロード不可。key remapping の実装方針を記載 |
| 6 | **Design decision** | Section 4.1: MonaAdapter の設計決定を明記。`"mona"` 選択時、factory は wrapper を作らず、既存 `Mona` クラスを直接返す。これにより checkpoint key が `adapter1.*`（nesting なし）となり、単純な文字列置換で key migration が可能になる |
| 7 | **Success criteria** | Section 7: 成功基準 #5 を追加「全 3 種の adapter で checkpoint save → load → forward pass の round-trip が成功する」 |

---

## Planning Quality Assessment

### $easy report

**ひとことで**: 実装規模は小さいが、checkpoint 後方互換と DualAttn のバグ修正がクリティカルパス。全タスクを Phase 1-3 に分割し、adapter 単体テストを先に通してから統合する順序でリスクを抑える。

**採点レビュー**:

| 案 | 採点軸 | 点数 | 根拠 |
|----|--------|------|------|
| Adapter selectable design | Product Fit | 5/5 | DINO branch のコア機能。adapter 比較実験の前提 |
| | Evidence Strength | 5/5 | 既存実装 (Mona) + 論文 (DANet) + spec 完備 |
| | User Value | 5/5 | adapter 切替で実験効率が大幅向上 |
| | Implementation Feasibility | 4/5 | 小規模な変更だが checkpoint migration に注意点あり |
| | Regression Safety | 4/5 | 既定値 "mona" で後方互換、ただし key 名変更で旧 checkpoint は要 remap |
| | Strategic Leverage | 4/5 | 今後の adapter 追加の基盤になる |
| | Security Safety | 5/5 | 権限・秘密情報・外部送信への影響なし |
| | Works In Practice | 4/5 | smoke test で検証可能、ただし backward compat 検証には GPU 時間が必要 |

**デグレ確認**:
- `team_validation_mode`: `native` (Plan agent による architecture review 実施)
- 仕様: `docs/superpowers/specs/2026-06-19-dino-adapter-selectable-design.md` に Spec delta 7 件を反映
- Plans.md: 本ファイル。Phase 1-3 の 11 task
- harness-mem / 記憶: 未設定 (`.claude/memory/` 未整備、prior decisions なし)
- TeamAgent / サブエージェント: Plan agent で architecture review 実施。Product / Security / QA / Skeptic 視点は native 検証
- product fit: DINO branch の adapter 比較実験に直結
- security: 影響なし（学習済みモデルの重み操作のみ、認証・認可・秘密情報・外部通信なし）
- works in practice: 各 task に DoD 明記。Phase 3 で smoke test + backward compat + checkpoint round-trip 検証
- formatter_baseline: `missing` → `skip_with_reason` (研究用 ML コードベース、既存 lint/formatter/CI なし、spec で要求なし)
- mirror / 配布: 影響なし
- test: テストフレームワークなし。smoke test は手動スクリプトで検証

---

## Phase 1: Adapter Package — 新規ファイル作成

Purpose: FC と Dual Attention の 2 種の adapter 実装 + factory を作成。Mona は既存 `mona.py` を直接再利用（wrapper 不要）。既存コードは一切変更しない。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | Create `dinosam/model/adapters/fc_adapter.py` — FCAdapter クラス | ファイルが存在し、FCAdapter(nn.Module) が spec Section 4.2 通りの構造を持つ（LayerNorm*γ + identity*γx → Linear(dim→bottleneck) → GELU → Dropout → Linear(bottleneck→dim) → residual）。forward(x, hw_shapes=None) が (B,N,C)→(B,N,C) を返す。gamma は 1e-6 初期化。`python -c "from dinosam.model.adapters.fc_adapter import FCAdapter; m=FCAdapter(768); print(m(torch.randn(2,4096,768)).shape)"` が成功すること | - | cc:TODO |
| 1.2 | Create `dinosam/model/adapters/dual_attn_adapter.py` — DualAttnAdapter + PAM_Module + CAM_Module | ファイルが存在し、3 クラスが spec Section 4.3 の構造を持つ。**要修正点を適用済み**: (a) PAM/CAM 内部の `+ x` residual を削除し、DualAttnAdapter.forward() で `out = (self.pam(x_2d) + self.cam(x_2d)) / 2 + x_2d` とする方式（二重加算バグ修正、局所変更で済むため）。代替として各 module から residual 除去も可（変更範囲が広い）。 (b) hw_shapes=None 時に `ValueError` を raise。 (c) factor パラメータを PAM_Module の query/key conv の reduction ratio に配線（`C // factor`）。gamma/beta は 0 初期化。`python -c "from dinosam.model.adapters.dual_attn_adapter import DualAttnAdapter; m=DualAttnAdapter(768); print(m(torch.randn(2,4096,768), (64,64)).shape)"` が成功すること | - | cc:TODO |
| 1.3 | Create `dinosam/model/adapters/__init__.py` — factory + exports | **(a)** `__init__.py` に `get_adapter(name, in_dim, factor=8)` 関数を実装。`"mona"` は `Mona` を直接返す（wrapper なし、nesting 回避）。`"fc"` は `FCAdapter`、`"dual_attn"` は `DualAttnAdapter` を返す。未知の name は ValueError。 **(b)** `python -c "from dinosam.model.adapters import get_adapter; [print(type(get_adapter(n,768))) for n in ['mona','fc','dual_attn']]"` が 3 種すべて成功すること | 1.1, 1.2 | cc:TODO |

---

## Phase 2: Integration — 既存ファイル変更

Purpose: encoder → model → CLI の順に adapter_type パラメータを配線する。Phase 1 完了が前提。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | Modify `dinosam/model/dinov3_encoder.py` — factory 統合 | `DINOv3MonaEncoder.__init__` が `adapter_type: str = "mona"` パラメータを受け取る。`from .adapters import get_adapter` を追加し、`from .mona import Mona` を**削除**。`self.mona1/mona2` を `self.adapter1/adapter2` に rename し、`get_adapter(adapter_type, embed_dim, factor=8)` で生成。`forward()` 内の `self.mona1/mona2` 呼び出しを `self.adapter1/adapter2` に更新。既定値 "mona" により既存の呼び出し元（depth_sam.py）は引数未指定でも動作継続 | Phase 1 | cc:TODO |
| 2.2 | Modify `dinosam/model/depth_sam.py` — adapter_type パラメータ追加 | `DepthSam.__init__` が `adapter_type: str = "mona"` パラメータを受け取り、`DINOv3MonaEncoder` に `adapter_type=adapter_type` として転送。`encoder_type != "dinov3"` 時は無視（dinov3_encoder 自体が None）。既存呼び出し元は既定値により動作継続 | 2.1 | cc:TODO |
| 2.3 | Modify `dinosam/train/trainer.py` — --adapter_type CLI 引数追加 | argparse に `--adapter_type` (type=str, default="mona", choices=["mona","fc","dual_attn"]) を追加。`_build_model_and_data()` 内の `DepthSam(...)` 呼び出しに `adapter_type=args.adapter_type` を追加。`--encoder sam` かつ `--adapter_type` が非既定値で明示指定された場合、warning を print（`"Warning: --adapter_type is ignored when --encoder sam"`）。`--encoder dinov3` 時のみ有効 | 2.2 | cc:TODO |
| 2.4 | Modify `dinosam/evaluate.py` — --adapter_type CLI 引数追加 | argparse に `--adapter_type` (type=str, default="mona", choices=["mona","fc","dual_attn"]) を追加。2 箇所の `DepthSam(...)` 呼び出し（phase1 用 L295、phase2 用 L318）に `adapter_type=args.adapter_type` を追加。`--encoder sam` かつ `--adapter_type` が非既定値で明示指定された場合、warning を print | 2.2 | cc:TODO |

---

## Phase 3: Compatibility & Verification

Purpose: checkpoint 後方互換の実装 + 全 adapter の動作検証。Phase 2 完了が前提。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | Implement legacy checkpoint key remapping | `_remap_legacy_state_dict()` 関数を実装。`model_state_dict` の key に `mona1.` プレフィックスがあれば `adapter1.` に、`mona2.` を `adapter2.` に文字列置換（wrapper なし設計により単純置換で済む）。**model_state_dict のみ remap し、optimizer_state_dict / scheduler_state_dict は触らない**。新しい形式（`adapter1.` プレフィックス）には何もしない（前方互換・冪等）。trainer.py の `_load_checkpoint()` と evaluate.py の `_load_model_state_dict()` の `load_state_dict` 呼び出し直前に適用 | Phase 2 | cc:TODO |
| 3.2 | Smoke test: adapter instantiation + forward pass shape check | スクリプト実行により 3 種の adapter がすべて import/instantiate/forward pass 可能であること。入力: `(B=2, N=4096, C=768)`, hw_shapes=(64,64)。出力 shape が入力と同じ `(2, 4096, 768)` であること。dual_attn は hw_shapes=None で ValueError が raise されること。fc は hw_shapes=None でも動作すること | Phase 1 | cc:TODO |
| 3.3 | Backward compatibility: mona adapter の出力一致検証 | 現在の DINO branch の checkpoint をロードし、`--adapter_type mona` で forward pass を実行。現在の hardcoded Mona 実装の出力と同一（allclose）であること。`_remap_legacy_state_dict` による旧 checkpoint ロードが成功すること | 3.1 | cc:TODO |
| 3.4 | Checkpoint round-trip: save/load for all adapter types | 3 種すべての adapter type で checkpoint を save → load → forward pass し、save 前と load 後の出力が一致すること（allclose）。これを trainer.py の checkpoint save/load 経由で確認すること | 3.1 | cc:TODO |

---

## Next Actions

新しいセッションの起動コマンド: `claude`
起動後の最初の入力: `/breezing all`
向いている場面: Phase 1 の adapter ファイル作成 (1.1, 1.2, 1.3) が並列実行可能で、Phase 2 以降は依存順に直列。チーム実行で Phase 1 を並列処理するのが効率的です
