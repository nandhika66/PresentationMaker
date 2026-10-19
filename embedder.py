# MiniLM embeddings WITHOUT PyTorch.
#
# Uses the official ONNX export of sentence-transformers/all-MiniLM-L6-v2
# and runs it with Microsoft's onnxruntime. It gives the same vectors as
# SentenceTransformer("all-MiniLM-L6-v2"): mean pooling + L2 normalization,
# max 256 tokens.
#
# Why: Windows Smart App Control / Application Control policies can block
# PyTorch's DLLs (WinError 4551). onnxruntime is signed by Microsoft.

import numpy as np

from config import EMBEDDING_MODEL, MAX_SEQ_LENGTH


class MiniLMEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL, max_length=MAX_SEQ_LENGTH):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        # Downloaded once (~90 MB), then reused from the Hugging Face cache
        model_path = hf_hub_download(model_name, "onnx/model.onnx")
        tokenizer_path = hf_hub_download(model_name, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def _encode_batch(self, texts):
        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        # First output = token embeddings, shape (batch, tokens, 384)
        token_embeddings = self.session.run(None, feeds)[0]

        # Mean pooling over real (non-padding) tokens
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        **_ignored,
    ):
        """Same call style as SentenceTransformer.encode()."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Batches")
            except ImportError:
                pass

        vectors = [
            self._encode_batch(list(texts[start:start + batch_size]))
            for start in batches
        ]
        embeddings = np.vstack(vectors).astype(np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

        return embeddings[0] if single else embeddings
