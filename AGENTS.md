# HF-RVC v1.0.0 開発環境セットアップ - 残作業指示書

Note: All code comments and documentation must be written in English.

## 🎯 現在の状況
- ✅ 基本環境セットアップ完了（uv、Python 3.13、依存関係）
- ✅ pytest実行環境構築完了（hydra-core修正済み）
- ✅ プロジェクト構造現代化完了（pyproject.toml移行）
- ✅ v1.0.0-dev ブランチでコミット済み
- ⚠️ 外部データセット依存テストエラー（Common Voice）
- ✅ Gradio UI static fixes completed (type errors resolved, runtime tests pending)
- ❌ RVCModel実装不完全（PyTorchモデル属性不足）

## 🔧 残り作業リスト

### 1. RVCModel実装の修正 【優先度: 高】
**問題**: `hf_rvc/models/modeling_rvc.py`のRVCModelクラスがPyTorchモデルの標準属性・メソッドを欠いている

**修正が必要な属性・メソッド**:
```python
# 不足している属性
- name_or_path: str
- device: torch.device
- dtype: torch.dtype
- training: bool

# 不足しているメソッド
- to_bettertransformer()
- eval()
- __call__() メソッド（モデル実行）
```

**作業**:
1. `hf_rvc/models/modeling_rvc.py`を開く
2. RVCModelクラスに不足している属性を追加
3. PyTorchモデルの標準メソッドを実装
4. HuggingFace Transformersの標準インターフェースに準拠

**✅ 修正完了後はこの項目を削除**

---

### 2. Gradio UI修正 【優先度: 高】
**問題**: `hf_rvc/tools/gradio_vc.py`に複数の型エラーとAPI変更対応が必要

**修正が必要な箇所**:
```python
# Gradio Audio APIの変更対応
source="microphone" → sources=["microphone"]
source="upload" → sources=["upload"]

# 型アノテーション修正
np.ndarray → np.ndarray[Any, np.dtype[Any]]
Audio = Tuple[int, np.ndarray] → Audio = Tuple[int, np.ndarray[Any, np.dtype[Any]]]

# torch.inference_mode()デコレータの型修正
```

**作業**:
1. `hf_rvc/tools/gradio_vc.py`を開く
2. Apply Gradio v4.x Audio API updates
3. Fix numpy type annotations
4. Resolve torch decorator type issues
5. Test the Gradio application at runtime
6. Ensure all code comments are in English

**✅ 修正完了後はこの項目を削除**

---

### 2.1 Pin Gradio version to bypass API changes 【優先度: 高】

**Problem**: Gradio v4.x API compatibility issues in `gradio_vc.py`

**Action**:
1. Pin the `gradio` dependency to `<4.0.0` in `pyproject.toml` or `requirements.txt`.
2. Run `uv pip install --upgrade gradio<4.0.0` to install the pinned version.

**✅ Pin complete and UI loads without API errors**

---

### 3. テストデータセット問題の修正 【優先度: 中】

**問題**: Common Voiceデータセットが`trust_remote_code=True`を要求してテストが失敗

**修正が必要なファイル**: `tests/models/test_feature_extraction_rvc.py`

**作業**:
1. テストファイルを開く
2. `load_dataset("common_voice", "ja", split="validation[:10%]")`に`trust_remote_code=True`を追加
3. または、外部データセット依存を削除してモックデータを使用
4. テスト実行で4つのエラーが解決されることを確認

**✅ 修正完了後はこの項目を削除**

---

### 4. 型チェックエラーの解決 【優先度: 中】
**問題**: mypy型チェックで多数のエラーが発生

**作業**:
1. `uv run mypy hf_rvc/`を実行してエラー確認
2. 不足している型スタブのインストール（types-torch、types-librosaなど）
3. 型アノテーションの追加・修正
4. mypy設定の調整（pyproject.tomlの[tool.mypy]セクション）

**✅ 修正完了後はこの項目を削除**

---

### 5. CLI機能の拡張テスト 【優先度: 低】
**問題**: 基本CLIは動作するが、音声変換機能のテストが未完了

**作業**:
1. `uv run hf-rvc convert --help`で利用可能なコマンドを確認
2. テスト用音声ファイルでの変換テスト
3. Gradio UI起動テスト（修正完了後）
4. リアルタイム音声変換のテスト

**✅ 修正完了後はこの項目を削除**

---

### 6. パフォーマンス最適化 【優先度: 低】
**問題**: 現在のコードカバレッジが14%と低い

**作業**:
1. 追加テストケースの作成
2. エッジケースのテスト追加
3. ベンチマークテストの実装
4. メモリ使用量とCPU/GPU使用率の最適化

**✅ 修正完了後はこの項目を削除**

---

## 🚀 修正手順

### 優先度1: RVCModel修正
```bash
# 1. モデルファイル確認
uv run python -c "from hf_rvc.models import RVCModel; print(dir(RVCModel))"

# 2. 修正後のテスト
uv run python -c "
from hf_rvc.models import RVCModel
model = RVCModel()
print(f'device: {model.device}')
print(f'dtype: {model.dtype}')
"
```

### 優先度2: Gradio修正
```bash
# 1. Gradioバージョン確認
uv run python -c "import gradio as gr; print(gr.__version__ if hasattr(gr, '__version__') else 'version unknown')"

# 2. 修正後のテスト
uv run python -c "
from hf_rvc.tools.gradio_vc import create_ui
# UIの基本インスタンス化テスト
"
```

### 最終確認
```bash
# 全テスト実行
uv run pytest tests/ -v --cov=hf_rvc --cov-report=term-missing

# 型チェック
uv run mypy hf_rvc/

# Linting
uv run ruff check .
```

## 📋 完了チェックリスト

- [ ] RVCModel PyTorch属性実装
- [ ] Gradio UI修正とテスト
- [ ] テストデータセット問題解決
- [ ] 型チェックエラー0個
- [ ] pytest全テスト成功
- [ ] CLI機能動作確認
- [ ] Gradio UI起動確認

## 🎯 目標
- テストカバレッジ80%以上
- mypy型チェック完全合格
- Gradio UI正常動作
- v1.0.0リリース準備完了

---
*このファイルは作業完了に伴い段階的に削除していきます*
