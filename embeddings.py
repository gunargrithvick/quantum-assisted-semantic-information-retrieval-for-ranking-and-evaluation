"""Sentence-transformer embeddings and cache validation."""
import hashlib
import json

import numpy as np
import torch

from config import EMBEDDING_CACHE_VERSION, EMBEDDING_MODEL_NAME, MODEL_ROOT

def _load_embedding_model(device=None):

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for training/search. "
            "Install the dependencies from requirements.txt."
        ) from exc

    kwargs={} if device is None else {"device":device}
    return SentenceTransformer(EMBEDDING_MODEL_NAME,**kwargs)



def _input_fingerprint(docs,labels=None):

    if labels is not None and len(docs)!=len(labels):
        raise ValueError("Documents and labels must have the same length")

    digest=hashlib.sha256()

    for index,doc in enumerate(docs):
        digest.update(str(index).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc).encode("utf-8"))
        digest.update(b"\0")

        if labels is not None:
            digest.update(repr(labels[index]).encode("utf-8"))
            digest.update(b"\0")

    return digest.hexdigest()



def generate_embeddings(docs,dataset_name,labels=None):

    if len(docs)==0:
        raise ValueError("Cannot generate embeddings for an empty document collection")

    if labels is not None and len(docs)!=len(labels):
        raise ValueError("Documents and labels must have the same length")

    cache_path=MODEL_ROOT / f"embeddings_{dataset_name}.npy"
    metadata_path=cache_path.with_suffix(".json")
    fingerprint=_input_fingerprint(docs,labels)

    if cache_path.exists() and metadata_path.exists():

        try:
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            embeddings=np.load(cache_path,allow_pickle=False)

            valid=(
                metadata.get("cache_version")==EMBEDDING_CACHE_VERSION and
                metadata.get("model_name")==EMBEDDING_MODEL_NAME and
                metadata.get("dataset_name")==dataset_name and
                metadata.get("doc_count")==len(docs) and
                metadata.get("input_fingerprint")==fingerprint and
                embeddings.ndim==2 and
                embeddings.shape[0]==len(docs) and
                np.isfinite(embeddings).all()
            )

            if valid:
                model=_load_embedding_model()
                return embeddings,model

            print("Embedding cache metadata does not match; regenerating.")

        except (OSError,ValueError,TypeError,json.JSONDecodeError) as exc:
            print(f"Invalid embedding cache ({exc}); regenerating.")

    device="cuda" if torch.cuda.is_available() else "cpu"

    model=_load_embedding_model(device)

    embeddings=np.asarray(model.encode(docs,batch_size=256,show_progress_bar=True))

    if embeddings.ndim!=2 or embeddings.shape[0]!=len(docs):
        raise RuntimeError("Embedding model returned an invalid shape")

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding model returned non-finite values")

    MODEL_ROOT.mkdir(parents=True,exist_ok=True)
    np.save(cache_path,embeddings)
    metadata_path.write_text(json.dumps({
        "cache_version":EMBEDDING_CACHE_VERSION,
        "model_name":EMBEDDING_MODEL_NAME,
        "dataset_name":dataset_name,
        "doc_count":len(docs),
        "embedding_dim":int(embeddings.shape[1]),
        "input_fingerprint":fingerprint,
    },indent=2),encoding="utf-8")

    return embeddings,model
