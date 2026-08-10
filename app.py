"""Application orchestration and interactive command-line interface."""
import os
import re
import hashlib
import io
import json
import pickle
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler,MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans


from config import (
    CANDIDATE_POOL_SIZE,
    CLASSICAL_RANK_WEIGHT,
    DATA_ROOT,
    EMBEDDING_CACHE_VERSION,
    EMBEDDING_MODEL_NAME,
    MODEL_ROOT,
    MODEL_STATE_VERSION,
    PAIRWISE_TRAINING_LIMIT,
    QSVM_RANK_WEIGHT,
    QUANTUM_RANK_WEIGHT,
    USE_HARD_CLUSTER_FILTER,
)
from data import (
    load_newsgroups,
    load_reuters,
    remove_cross_split_duplicates,
    split_documents,
)
from embeddings import _input_fingerprint,generate_embeddings
from metrics import (
    build_eval_queries,
    compute_map,
    compute_ndcg,
    normalize_scores,
    precision_at_k,
    recall_at_k,
)
from plotting import display_current_plot
from quantum import (
    analyze_features,
    apply_feature_weights,
    apply_pca,
    build_faiss_index,
    normalize_embeddings,
    sample_training_indices,
    _build_quantum_kernel,
)
from text import clean_text,format_snippet


docs=None
labels=None
doc_topics=None
eval_docs=None
eval_labels=None
eval_topics=None
embeddings=None
model=None
index=None
current_dataset=None
feature_weights=None

cluster_pca=None
cluster_scaler=None
kmeans_model=None
cluster_labels=None

qsvm_model=None
qsvm_pca=None
qsvm_scaler=None
qsvm_kernel=None
qsvm_train_reduced=None
qsvm_label_binarizer=None
qsvm_multilabel=False
ranker_model=None
ranker_scaler=None
ranker_kernel=None
ranker_train_features=None
DEFAULT_RANKING_WEIGHTS=(
    CLASSICAL_RANK_WEIGHT,
    QUANTUM_RANK_WEIGHT,
    QSVM_RANK_WEIGHT,
)
ranking_weights=DEFAULT_RANKING_WEIGHTS

dataset_results={}
comparison_results={}
candidate_recall_results={}
multi_seed_results={}
query_embedding_cache={}
quantum_score_cache={}
tfidf_vectorizer=None
tfidf_matrix=None


def cluster_documents(embeddings,n_clusters=10):

    global cluster_pca,cluster_scaler

    reduced,cluster_pca,cluster_scaler=apply_pca(embeddings)

    if len(reduced)==0:
        raise ValueError("Clustering requires at least one document")

    n_clusters=min(n_clusters,len(reduced))
    kmeans=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)

    cluster_labels=kmeans.fit_predict(reduced)

    return kmeans,cluster_labels


def train_qsvm(embeddings,labels,topic_sets=None):

    global qsvm_model,qsvm_pca,qsvm_scaler,qsvm_kernel,qsvm_train_reduced
    global qsvm_label_binarizer,qsvm_multilabel

    if len(embeddings)<2 or len(np.unique(labels))<2:
        raise ValueError("QSVM training requires at least two samples and two classes")

    reduced,qsvm_pca,qsvm_scaler=apply_pca(embeddings)
    qsvm_train_reduced=reduced

    qsvm_kernel=_build_quantum_kernel(reduced.shape[1])

    kernel_matrix = qsvm_kernel.evaluate(reduced.tolist(), reduced.tolist())

    qsvm_label_binarizer=None
    qsvm_multilabel=topic_sets is not None

    if qsvm_multilabel:
        if len(topic_sets)!=len(labels) or any(len(topics)==0 for topics in topic_sets):
            raise ValueError("Every document must have at least one topic")

        qsvm_label_binarizer=MultiLabelBinarizer()
        classifier_labels=qsvm_label_binarizer.fit_transform(topic_sets)

        if classifier_labels.shape[1]<2:
            raise ValueError("Multi-label QSVM training requires at least two topics")

        qsvm_model=OneVsRestClassifier(
            SVC(kernel="precomputed",decision_function_shape="ovr")
        )
    else:
        classifier_labels=labels
        qsvm_model=SVC(kernel="precomputed",decision_function_shape="ovr")

    qsvm_model.fit(kernel_matrix,classifier_labels)


def build_query_document_features(query_embeddings,document_embeddings):

    """Create compact relevance features for a query-document pair.

    These features are deliberately query-dependent. The quantum classifier
    therefore learns relevance of a document to a query rather than only
    predicting the document's category.
    """

    queries=normalize_embeddings(np.asarray(query_embeddings,dtype=float))
    documents=normalize_embeddings(np.asarray(document_embeddings,dtype=float))

    if len(queries)!=1:
        raise ValueError("Exactly one query embedding is required")

    if len(documents)==0:
        return np.empty((0,4),dtype=float)

    difference=np.abs(documents-queries[0])
    cosine=np.clip(documents @ queries[0],-1.0,1.0)
    euclidean=np.linalg.norm(documents-queries[0],axis=1)

    return np.column_stack((
        cosine,
        1.0/(1.0+euclidean),
        1.0-np.mean(difference,axis=1),
        1.0-np.max(difference,axis=1),
    )).astype(np.float64)


def _pair_is_relevant(query_index,document_index,labels,topic_sets=None):

    if topic_sets is None:
        return labels[query_index]==labels[document_index]

    return bool(
        set(topic_sets[query_index]).intersection(topic_sets[document_index])
    )


def train_quantum_relevance_ranker(
    train_embeddings,train_labels,train_docs,topic_sets=None,max_pairs=PAIRWISE_TRAINING_LIMIT
):

    """Train a simulator-backed quantum-kernel relevance classifier.

    Positive and negative query-document pairs are sampled from training data
    only. The pair budget keeps statevector-kernel simulation practical while
    preserving balanced supervision.
    """

    global ranker_model,ranker_scaler,ranker_kernel,ranker_train_features

    values=np.asarray(train_embeddings)
    labels=list(train_labels)

    if len(values)!=len(labels) or len(values)!=len(train_docs):
        raise ValueError("Pairwise ranker inputs must have matching lengths")

    if len(values)<4:
        raise ValueError("Pairwise quantum ranking requires at least four documents")

    rng=np.random.RandomState(43)
    query_count=min(40,len(values))
    query_indices=sample_training_indices(labels,query_count,seed=43)
    per_side=max(1,max_pairs//(2*max(1,len(query_indices))))

    pair_features=[]
    pair_labels=[]

    for query_index in query_indices:
        positives=[]
        negatives=[]

        for document_index in range(len(values)):
            if document_index==query_index:
                continue
            if _pair_is_relevant(query_index,document_index,labels,topic_sets):
                positives.append(document_index)
            else:
                negatives.append(document_index)

        if not positives or not negatives:
            continue

        positive_indices=rng.choice(
            positives,size=min(per_side,len(positives)),replace=False
        )
        negative_indices=rng.choice(
            negatives,size=min(per_side,len(negatives)),replace=False
        )

        for document_index in positive_indices:
            pair_features.append(
                build_query_document_features(
                    values[query_index:query_index+1],
                    values[document_index:document_index+1],
                )[0]
            )
            pair_labels.append(1)

        for document_index in negative_indices:
            pair_features.append(
                build_query_document_features(
                    values[query_index:query_index+1],
                    values[document_index:document_index+1],
                )[0]
            )
            pair_labels.append(0)

    if len(pair_features)<4 or len(set(pair_labels))<2:
        raise ValueError("Could not create balanced query-document training pairs")

    pair_features=np.asarray(pair_features,dtype=float)
    ranker_scaler=MinMaxScaler()
    ranker_train_features=ranker_scaler.fit_transform(pair_features)
    ranker_kernel=_build_quantum_kernel(ranker_train_features.shape[1])
    kernel_matrix=ranker_kernel.evaluate(
        ranker_train_features.tolist(),ranker_train_features.tolist()
    )
    ranker_model=SVC(
        kernel="precomputed",class_weight="balanced",C=2.0,
        decision_function_shape="ovr",
    )
    ranker_model.fit(kernel_matrix,np.asarray(pair_labels,dtype=int))

    return len(pair_labels)


def model_state_path(dataset_name):

    return MODEL_ROOT / f"model_{dataset_name}_train.pkl"


class _RestrictedUnpickler(pickle.Unpickler):

    """Allow only classes needed by locally saved model pipelines."""

    _allowed_globals={
        ("builtins","bool"),
        ("builtins","bytes"),
        ("builtins","complex"),
        ("builtins","dict"),
        ("builtins","float"),
        ("builtins","frozenset"),
        ("builtins","int"),
        ("builtins","list"),
        ("builtins","object"),
        ("builtins","set"),
        ("builtins","slice"),
        ("builtins","str"),
        ("collections","OrderedDict"),
        ("copyreg","_reconstructor"),
        ("numpy","dtype"),
        ("numpy","ndarray"),
        ("numpy.core.multiarray","_reconstruct"),
        ("numpy.core.multiarray","scalar"),
        ("numpy.core.numeric","_frombuffer"),
        ("numpy._core.multiarray","_reconstruct"),
        ("numpy._core.multiarray","scalar"),
        ("numpy._core.numeric","_frombuffer"),
        ("sklearn.cluster._kmeans","KMeans"),
        ("sklearn.decomposition._pca","PCA"),
        ("sklearn.multiclass","OneVsRestClassifier"),
        ("sklearn.preprocessing._data","StandardScaler"),
        ("sklearn.preprocessing._data","MinMaxScaler"),
        ("sklearn.preprocessing._label","LabelBinarizer"),
        ("sklearn.preprocessing._label","MultiLabelBinarizer"),
        ("sklearn.svm._classes","SVC"),
    }

    def find_class(self,module,name):

        if (module,name) not in self._allowed_globals:
            raise pickle.UnpicklingError(
                f"Blocked model-state class: {module}.{name}"
            )

        return super().find_class(module,name)


def load_model_state(path):

    """Load a locally generated state without permitting arbitrary globals."""

    with path.open("rb") as f:
        return _RestrictedUnpickler(io.BytesIO(f.read())).load()


_RUNTIME_STATE_NAMES=(
    "docs","labels","doc_topics","eval_docs","eval_labels","eval_topics",
    "embeddings","model","index","current_dataset","feature_weights",
    "cluster_pca","cluster_scaler","kmeans_model","cluster_labels",
    "qsvm_model","qsvm_pca","qsvm_scaler","qsvm_kernel",
    "qsvm_train_reduced","qsvm_label_binarizer","qsvm_multilabel",
    "ranker_model","ranker_scaler","ranker_kernel","ranker_train_features",
    "tfidf_vectorizer","tfidf_matrix",
    "ranking_weights",
)


def _transactional_pipeline(function):

    """Restore the last usable pipeline if a replacement operation fails."""

    @wraps(function)
    def wrapped(*args,**kwargs):
        previous_state={
            name:globals()[name]
            for name in _RUNTIME_STATE_NAMES
        }

        try:
            return function(*args,**kwargs)
        except Exception:
            globals().update(previous_state)
            raise

    return wrapped


def save_model_state(dataset_name,docs,labels):

    state_path=model_state_path(dataset_name)

    required=(
        feature_weights is not None and
        cluster_pca is not None and
        cluster_scaler is not None and
        kmeans_model is not None and
        cluster_labels is not None and
        qsvm_model is not None and
        qsvm_pca is not None and
        qsvm_scaler is not None and
        qsvm_train_reduced is not None and
        ranker_model is not None and
        ranker_scaler is not None and
        ranker_train_features is not None
    )

    if not required:
        raise RuntimeError("Cannot save an incomplete trained pipeline")

    state={
        "state_version":MODEL_STATE_VERSION,
        "dataset_name":dataset_name,
        "input_fingerprint":_input_fingerprint(docs,labels),
        "feature_weights":feature_weights,
        "cluster_pca":cluster_pca,
        "cluster_scaler":cluster_scaler,
        "kmeans_model":kmeans_model,
        "cluster_labels":np.asarray(cluster_labels),
        "qsvm_model":qsvm_model,
        "qsvm_pca":qsvm_pca,
        "qsvm_scaler":qsvm_scaler,
        "qsvm_train_reduced":np.asarray(qsvm_train_reduced),
        "qsvm_label_binarizer":qsvm_label_binarizer,
        "qsvm_multilabel":qsvm_multilabel,
        "ranker_model":ranker_model,
        "ranker_scaler":ranker_scaler,
        "ranker_train_features":np.asarray(ranker_train_features),
        "ranking_weights":tuple(ranking_weights),
    }

    MODEL_ROOT.mkdir(parents=True,exist_ok=True)
    temporary_path=state_path.with_suffix(".tmp")
    temporary_path.write_bytes(pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL))
    temporary_path.replace(state_path)


def restore_model_state(dataset_name,docs,labels,raw_embeddings):

    global embeddings,feature_weights
    global cluster_pca,cluster_scaler,kmeans_model,cluster_labels
    global qsvm_model,qsvm_pca,qsvm_scaler,qsvm_kernel,qsvm_train_reduced,index
    global qsvm_label_binarizer,qsvm_multilabel
    global ranker_model,ranker_scaler,ranker_kernel,ranker_train_features
    global ranking_weights

    state_path=model_state_path(dataset_name)

    if not state_path.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {state_path}. Train this dataset first."
        )

    state=load_model_state(state_path)

    required_keys={
        "state_version","dataset_name","input_fingerprint","feature_weights",
        "cluster_pca","cluster_scaler","kmeans_model","cluster_labels",
        "qsvm_model","qsvm_pca","qsvm_scaler","qsvm_train_reduced",
        "qsvm_label_binarizer","qsvm_multilabel",
        "ranker_model","ranker_scaler","ranker_train_features",
    }

    if not required_keys.issubset(state):
        raise RuntimeError("Saved pipeline is incomplete")

    if state["state_version"]!=MODEL_STATE_VERSION or state["dataset_name"]!=dataset_name:
        raise RuntimeError("Saved pipeline version or dataset does not match")

    if state["input_fingerprint"]!=_input_fingerprint(docs,labels):
        raise RuntimeError("Saved pipeline does not match the current dataset")

    raw_embeddings=np.asarray(raw_embeddings)
    feature_weights=np.asarray(state["feature_weights"])

    if raw_embeddings.ndim!=2 or len(raw_embeddings)!=len(docs):
        raise RuntimeError("Cached embeddings do not match the current documents")

    if feature_weights.shape!=(raw_embeddings.shape[1],):
        raise RuntimeError("Saved feature weights do not match embedding dimensions")

    embeddings=apply_feature_weights(raw_embeddings,feature_weights)
    cluster_pca=state["cluster_pca"]
    cluster_scaler=state["cluster_scaler"]
    kmeans_model=state["kmeans_model"]
    cluster_labels=np.asarray(state["cluster_labels"])
    qsvm_model=state["qsvm_model"]
    qsvm_pca=state["qsvm_pca"]
    qsvm_scaler=state["qsvm_scaler"]
    qsvm_train_reduced=np.asarray(state["qsvm_train_reduced"])
    qsvm_label_binarizer=state["qsvm_label_binarizer"]
    qsvm_multilabel=bool(state["qsvm_multilabel"])
    ranker_model=state["ranker_model"]
    ranker_scaler=state["ranker_scaler"]
    ranker_train_features=np.asarray(state["ranker_train_features"])
    ranking_weights=_validate_ranking_weights(
        state.get("ranking_weights",DEFAULT_RANKING_WEIGHTS)
    )

    if len(cluster_labels)!=len(embeddings):
        raise RuntimeError("Saved cluster labels do not match embeddings")

    qsvm_kernel=_build_quantum_kernel(qsvm_train_reduced.shape[1])
    ranker_kernel=_build_quantum_kernel(ranker_train_features.shape[1])
    index=build_faiss_index(embeddings)
    _build_tfidf_index()
    clear_ranking_caches()


def clear_ranking_caches():

    """Discard query-dependent state after a pipeline changes."""

    query_embedding_cache.clear()
    quantum_score_cache.clear()


def _build_tfidf_index():

    """Build a local lexical candidate index from the indexed documents."""

    global tfidf_vectorizer,tfidf_matrix

    if docs is None or len(docs)==0:
        tfidf_vectorizer=None
        tfidf_matrix=None
        return

    tfidf_vectorizer=TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        sublinear_tf=True,
        max_features=100000,
    )
    tfidf_matrix=tfidf_vectorizer.fit_transform(
        clean_text(document) for document in docs
    )


def _encode_query(query):

    global model,feature_weights

    cleaned_query=clean_text(query)

    if not cleaned_query:
        raise ValueError("Search query must contain at least one non-whitespace character")

    if model is None:
        raise RuntimeError("Embedding model is unavailable. Train or load a model first.")

    cached=query_embedding_cache.get(cleaned_query)

    if cached is not None:
        return cached.copy()

    q_embed=apply_feature_weights(model.encode([cleaned_query]),feature_weights)
    query_embedding_cache[cleaned_query]=np.asarray(q_embed).copy()
    return q_embed


def _tfidf_candidate_indices(query,k):

    if tfidf_vectorizer is None or tfidf_matrix is None or k<=0:
        return np.array([],dtype=int),np.array([],dtype=float)

    cleaned_query=clean_text(query)

    if not cleaned_query:
        raise ValueError("Search query must contain at least one non-whitespace character")

    query_vector=tfidf_vectorizer.transform([cleaned_query])
    scores=np.asarray((tfidf_matrix @ query_vector.T).toarray()).reshape(-1)
    ranked=np.argsort(-scores,kind="stable")[:min(k,len(scores))]

    return ranked,scores


def retrieve_candidates(query,k=CANDIDATE_POOL_SIZE):

    global model,index,feature_weights
    global cluster_pca,cluster_scaler,kmeans_model,cluster_labels

    q_embed=_encode_query(query)

    if index.ntotal==0 or k<=0:
        return np.array([],dtype=int),q_embed

    search_vector=normalize_embeddings(q_embed)
    cluster_ready=(
        kmeans_model is not None and
        cluster_labels is not None and
        cluster_pca is not None and
        cluster_scaler is not None and
        len(cluster_labels)==index.ntotal
    )

    if USE_HARD_CLUSTER_FILTER and cluster_ready:
        reduced_query=cluster_pca.transform(cluster_scaler.transform(q_embed))
        cluster_id=kmeans_model.predict(reduced_query)[0]
        cluster_docs=np.flatnonzero(cluster_labels==cluster_id)
    else:
        cluster_docs=None

    if cluster_docs is None or len(cluster_docs)==0:
        search_k=min(k,index.ntotal)
        distances,indices=index.search(search_vector,search_k)
        semantic_candidates=indices[0]
    else:
        # IndexFlat search is already exhaustive. Request all results so a
        # relevant member outside the global top-k is not discarded before
        # cluster filtering.
        distances,indices=index.search(search_vector,index.ntotal)
        all_candidates=indices[0]
        semantic_candidates=all_candidates[np.isin(all_candidates,cluster_docs)][:k]

    if USE_HARD_CLUSTER_FILTER:
        candidates=semantic_candidates
    else:
        lexical_candidates,_=_tfidf_candidate_indices(query,k)
        merged=np.unique(np.concatenate((semantic_candidates,lexical_candidates)))
        ranks={}

        for rank,document_index in enumerate(semantic_candidates):
            ranks[int(document_index)]=ranks.get(int(document_index),0.0)+1.0/(60+rank+1)

        for rank,document_index in enumerate(lexical_candidates):
            ranks[int(document_index)]=ranks.get(int(document_index),0.0)+1.0/(60+rank+1)

        candidates=np.asarray(
            sorted(merged,key=lambda value:(-ranks[int(value)],int(value)))[:k],
            dtype=int,
        )

    candidates=candidates[candidates>=0]

    return np.array(candidates),q_embed


def _classical_candidate_scores(query,candidates=None):

    if candidates is None:
        candidates,q_embed=retrieve_candidates(query)
    else:
        candidates=np.asarray(candidates,dtype=int)
        q_embed=_encode_query(query)

    if len(candidates)==0:
        return candidates,q_embed,np.array([],dtype=float)

    candidate_embeddings=embeddings[candidates]
    classical_scores=normalize_scores(
        cosine_similarity(q_embed,candidate_embeddings)[0]
    )

    return candidates,q_embed,classical_scores


def classical_ranking(query,candidates=None):

    """Rank a shared candidate set using semantic similarity only."""

    candidates,_,classical_scores=_classical_candidate_scores(query,candidates)

    if len(candidates)==0:
        return candidates

    return candidates[np.argsort(classical_scores)[::-1]]


def lexical_baseline_ranking(query,k=25,candidates=None):

    """Rank documents by cleaned-token overlap.

    When ``candidates`` is supplied, only that shared candidate pool is
    scored. This is the fair comparison mode used during evaluation.
    """

    global docs

    if docs is None:
        raise RuntimeError("Documents are unavailable. Train or load a model first.")

    if k<=0:
        raise ValueError("k must be positive")

    query_terms=set(clean_text(query).split())

    if not query_terms:
        raise ValueError("Search query must contain at least one non-whitespace character")

    overlap_scores=np.fromiter(
        (
            len(query_terms.intersection(set(clean_text(document).split())))
            for document in docs
        ),
        dtype=float,
        count=len(docs),
    )

    ranked=np.argsort(-overlap_scores,kind="stable")

    if candidates is not None:
        candidate_set=np.asarray(candidates,dtype=int)
        candidate_set=candidate_set[(candidate_set>=0)&(candidate_set<len(docs))]
        ranked=candidate_set[np.argsort(-overlap_scores[candidate_set],kind="stable")]

    return ranked[:min(k,len(ranked))]


def tfidf_baseline_ranking(query,k=25,candidates=None):

    """Rank documents with TF-IDF cosine similarity."""

    if docs is None:
        raise RuntimeError("Documents are unavailable. Train or load a model first.")

    if k<=0:
        raise ValueError("k must be positive")

    ranked,scores=_tfidf_candidate_indices(query,len(docs))

    if candidates is not None:
        candidate_set=np.asarray(candidates,dtype=int)
        candidate_set=candidate_set[(candidate_set>=0)&(candidate_set<len(docs))]
        ranked=candidate_set[np.argsort(-scores[candidate_set],kind="stable")]

    return ranked[:min(k,len(ranked))]


def _validate_ranking_weights(weights):

    values=np.asarray(weights,dtype=float)

    if values.shape!=(3,) or not np.isfinite(values).all():
        raise ValueError("Ranking weights must contain three finite values")

    if np.any(values<0) or not np.isclose(values.sum(),1.0):
        raise ValueError("Ranking weights must be non-negative and sum to one")

    return tuple(float(value) for value in values)


def _quantum_component_scores(query,candidates=None):

    global embeddings, qsvm_pca, qsvm_scaler, qsvm_kernel, qsvm_model
    global qsvm_train_reduced, qsvm_multilabel
    global ranker_scaler,ranker_kernel,ranker_model,ranker_train_features

    candidates,q_embed,classical_scores=_classical_candidate_scores(query,candidates)
    cache_key=(clean_text(query),tuple(int(value) for value in candidates))
    cached=quantum_score_cache.get(cache_key)

    if cached is not None:
        cached_candidates,cached_components=cached
        return (
            cached_candidates.copy(),
            tuple(component.copy() for component in cached_components),
        )

    if len(candidates)==0:
        empty=np.array([],dtype=float)
        return candidates,(empty,empty,empty)

    candidate_embeddings=embeddings[candidates]

    quantum_state_ready=(
        qsvm_pca is not None and
        qsvm_scaler is not None and
        qsvm_kernel is not None and
        qsvm_train_reduced is not None and
        ranker_scaler is not None and
        ranker_kernel is not None and
        ranker_model is not None and
        ranker_train_features is not None
    )

    if not quantum_state_ready:
        raise RuntimeError(
            "Quantum ranking state is unavailable. Train or load a quantum pipeline first."
        )

    pair_features=build_query_document_features(q_embed,candidate_embeddings)
    pair_features=ranker_scaler.transform(pair_features)
    pair_kernel=ranker_kernel.evaluate(
        pair_features.tolist(),ranker_train_features.tolist()
    )
    quantum_scores=normalize_scores(
        np.asarray(ranker_model.decision_function(pair_kernel)).reshape(-1)
    )

    reduced_docs=qsvm_pca.transform(qsvm_scaler.transform(candidate_embeddings))
    reduced_query=qsvm_pca.transform(qsvm_scaler.transform(q_embed))

    if qsvm_model is None:
        qsvm_scores=np.zeros(len(candidate_embeddings))
    else:
        query_train_kernel=qsvm_kernel.evaluate(
            reduced_query.tolist(),qsvm_train_reduced.tolist()
        )
        candidate_train_kernel=qsvm_kernel.evaluate(
            reduced_docs.tolist(),qsvm_train_reduced.tolist()
        )

        decision_scores=np.asarray(
            qsvm_model.decision_function(candidate_train_kernel)
        )

        if qsvm_multilabel:
            query_topic_scores=np.asarray(
                qsvm_model.decision_function(query_train_kernel)
            )
            query_topic_weights=normalize_scores(query_topic_scores[0])
            qsvm_scores=decision_scores @ query_topic_weights
        else:
            predicted_class=qsvm_model.predict(query_train_kernel)[0]
            classes=np.asarray(qsvm_model.classes_)

            if decision_scores.ndim==1:
                positive_class=classes[-1]
                qsvm_scores=(
                    decision_scores
                    if predicted_class==positive_class
                    else -decision_scores
                )
            else:
                class_indices=np.flatnonzero(classes==predicted_class)
                qsvm_scores=(
                    decision_scores[:,class_indices[0]]
                    if len(class_indices)>0
                    else np.zeros(len(candidate_embeddings))
                )

        qsvm_scores=normalize_scores(qsvm_scores)

    components=(classical_scores,quantum_scores,qsvm_scores)

    if len(quantum_score_cache)>=512:
        quantum_score_cache.pop(next(iter(quantum_score_cache)))

    quantum_score_cache[cache_key]=(
        candidates.copy(),tuple(component.copy() for component in components)
    )

    return candidates,components


def hybrid_ranking(query,weights=None,candidates=None):

    if weights is None:
        weights=ranking_weights

    classical_weight,quantum_weight,qsvm_weight=_validate_ranking_weights(weights)
    candidates,components=_quantum_component_scores(query,candidates)

    if len(candidates)==0:
        return candidates

    classical_scores,quantum_scores,qsvm_scores=components
    final_scores=(
        classical_weight*classical_scores+
        quantum_weight*quantum_scores+
        qsvm_weight*qsvm_scores
    )

    return candidates[np.argsort(final_scores)[::-1]]


def _ranking_weight_grid():

    candidates=[]

    for classical_units in range(0,11):
        for quantum_units in range(0,11-classical_units):
            qsvm_units=10-classical_units-quantum_units
            if quantum_units+qsvm_units>=classical_units:
                candidates.append((
                    classical_units/10,
                    quantum_units/10,
                    qsvm_units/10,
                ))

    return candidates


def tune_ranking_weights(query_docs,query_labels,corpus_labels,
                         query_topics=None,corpus_topics=None,
                         n_per_class=1):

    """Choose fusion weights on validation queries only.

    MAP@10 is the primary selection objective, followed by NDCG@10 and
    Precision@10 as deterministic tie-breakers. The caller must provide a
    corpus that excludes the validation documents.
    """

    global ranking_weights

    eval_queries=build_eval_queries(
        query_docs,query_labels,corpus_labels,
        query_topics=query_topics,corpus_topics=corpus_topics,
        n_per_class=n_per_class,
    )

    if len(eval_queries)==0:
        raise ValueError("Could not build validation queries for weight tuning")

    prepared=[]

    for query in eval_queries:
        candidates,components=_quantum_component_scores(query["query"])
        prepared.append((query,candidates,components))

    best_weights=None
    best_objective=None

    for weights in _ranking_weight_grid():
        rows=[]

        for query,candidates,components in prepared:
            classical_scores,quantum_scores,qsvm_scores=components
            final_scores=(
                weights[0]*classical_scores+
                weights[1]*quantum_scores+
                weights[2]*qsvm_scores
            )
            ranked=candidates[np.argsort(final_scores)[::-1]]
            relevant=query["relevant"]
            rows.append((
                precision_at_k(ranked,relevant,5),
                precision_at_k(ranked,relevant,10),
                compute_map(ranked,relevant),
                compute_ndcg(ranked,relevant),
            ))

        summary=np.mean(np.asarray(rows,dtype=float),axis=0)
        objective=(summary[2],summary[3],summary[1])

        if best_objective is None or objective>best_objective:
            best_objective=objective
            best_weights=weights

    ranking_weights=_validate_ranking_weights(best_weights)
    return ranking_weights,best_objective,len(eval_queries)


def _evaluate_queries(eval_queries,verbose=False,reveal_queries=False):

    ranking_functions=(
        ("Simple lexical baseline",lexical_baseline_ranking),
        ("TF-IDF baseline",tfidf_baseline_ranking),
        ("Classical semantic",classical_ranking),
        ("Quantum-assisted",hybrid_ranking),
    )
    results_by_method={name:[] for name,_ in ranking_functions}
    candidate_recalls=[]

    for q in eval_queries:
        relevant=q["relevant"]
        candidate_pool,_=retrieve_candidates(q["query"],k=CANDIDATE_POOL_SIZE)
        candidate_recalls.append(
            recall_at_k(candidate_pool,relevant,CANDIDATE_POOL_SIZE)
        )

        for method_name,ranking_function in ranking_functions:
            ranked=ranking_function(q["query"],candidates=candidate_pool)
            results_by_method[method_name].append((
                precision_at_k(ranked,relevant,5),
                precision_at_k(ranked,relevant,10),
                compute_map(ranked,relevant),
                compute_ndcg(ranked,relevant),
            ))

        if verbose:
            query_display=(
                f" \"{format_snippet(q['query'],60)}\""
                if reveal_queries else ""
            )
            p5,p10,map_score,ndcg_score=results_by_method["Quantum-assisted"][-1]
            print(f"Query source {q['source_idx']} (label {q['label']}){query_display}")
            print(f"  P@5={p5:.3f}  P@10={p10:.3f}  MAP={map_score:.3f}  NDCG={ndcg_score:.3f}")

    return results_by_method,candidate_recalls


def evaluate_model(n_per_class=2,verbose=True,reveal_queries=False,query_seed=42):

    # Runs Precision@5, Precision@10, MAP, and NDCG against the ground truth
    # from build_eval_queries. Relevance here is decided by the dataset's
    # own category labels, independently of what the model retrieves â€”
    # unlike the old approach, which inferred "relevant" from the model's
    # own top results and then evaluated the model against that.

    global docs,labels,doc_topics,eval_docs,eval_labels,eval_topics,current_dataset
    global candidate_recall_results

    if docs is None or labels is None or eval_docs is None or eval_labels is None:
        print("Train or load a model first.")
        return

    dataset_results.pop(current_dataset,None)
    comparison_results.pop(current_dataset,None)

    eval_queries=build_eval_queries(
        eval_docs,eval_labels,labels,
        query_topics=eval_topics,corpus_topics=doc_topics,
        n_per_class=n_per_class,seed=query_seed,
    )

    if len(eval_queries)==0:
        print("Could not build any evaluation queries (classes too small).")
        return

    results_by_method,candidate_recalls=_evaluate_queries(
        eval_queries,verbose=verbose,reveal_queries=reveal_queries
    )

    comparison_results[current_dataset]={
        method:np.asarray(values,dtype=float).tolist()
        for method,values in results_by_method.items()
    }
    dataset_results[current_dataset]=comparison_results[current_dataset]["Quantum-assisted"]
    candidate_recall_results[current_dataset]=candidate_recalls

    print("\n--- Evaluation summary (held-out queries,",len(eval_queries),"queries) ---")
    print("Dataset:",current_dataset)
    print(f"Candidate Recall@{CANDIDATE_POOL_SIZE}: {np.mean(candidate_recalls):.4f}")
    print("\n{:<24} {:>12} {:>12} {:>12} {:>12}".format(
        "Method","Precision@5","Precision@10","MAP@10","NDCG@10"
    ))
    print("-"*76)

    for method,values in results_by_method.items():
        summary=np.mean(np.asarray(values,dtype=float),axis=0)
        print("{:<24} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            method,summary[0],summary[1],summary[2],summary[3]
        ))

    return results_by_method,candidate_recalls


def evaluate_across_seeds(seeds=(42,43,44),n_per_class=2):

    """Report mean and variation across fixed held-out query selections."""

    global multi_seed_results

    if docs is None or labels is None or eval_docs is None or eval_labels is None:
        raise RuntimeError("Train or load a model first.")

    seeds=tuple(int(seed) for seed in seeds)

    if len(seeds)<2:
        raise ValueError("Provide at least two fixed seeds")

    per_seed=[]

    for seed in seeds:
        eval_queries=build_eval_queries(
            eval_docs,eval_labels,labels,
            query_topics=eval_topics,corpus_topics=doc_topics,
            n_per_class=n_per_class,seed=seed,
        )

        if not eval_queries:
            raise ValueError(f"Seed {seed} produced no evaluation queries")

        methods,candidate_recalls=_evaluate_queries(eval_queries)
        per_seed.append((methods,candidate_recalls,len(eval_queries)))

    method_names=per_seed[0][0].keys()
    summary={}

    print("\n--- Multi-seed evaluation (held-out queries) ---")
    print("Dataset:",current_dataset,"Seeds:",", ".join(map(str,seeds)))
    print("{:<24} {:>12} {:>12} {:>12} {:>12}".format(
        "Method","P@10 mean+/-std","MAP@10 mean+/-std","NDCG@10 mean+/-std","Recall@100"
    ))
    print("-"*108)

    for method in method_names:
        metric_means=np.asarray([
            np.mean(np.asarray(methods[method],dtype=float),axis=0)
            for methods,_,_ in per_seed
        ])
        recall_values=np.asarray([
            np.mean(candidate_recalls) for _,candidate_recalls,_ in per_seed
        ])
        summary[method]={
            "mean":metric_means.mean(axis=0).tolist(),
            "std":metric_means.std(axis=0,ddof=0).tolist(),
            "candidate_recall_mean":float(recall_values.mean()),
            "candidate_recall_std":float(recall_values.std(ddof=0)),
        }
        mean=summary[method]["mean"]
        std=summary[method]["std"]
        print(
            "{:<24} {:>5.4f}+/-{:<5.4f} {:>5.4f}+/-{:<5.4f} {:>5.4f}+/-{:<5.4f} {:>5.4f}".format(
                method,mean[1],std[1],mean[2],std[2],mean[3],std[3],
                summary[method]["candidate_recall_mean"],
            )
        )

    multi_seed_results[current_dataset]={
        "seeds":list(seeds),"queries_per_seed":[count for _,_,count in per_seed],
        "methods":summary,
    }
    return multi_seed_results[current_dataset]


def show_metrics():

    global embeddings,comparison_results

    if len(dataset_results)==0 and len(comparison_results)==0:
        print("No evaluation results yet.")
        return

    try:
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "seaborn is required to show evaluation graphs. "
            "Install the dependencies from requirements.txt."
        ) from exc

    if embeddings is not None:
        analyze_features(embeddings,show_plot=True)

    if comparison_results:
        source=comparison_results
    else:
        source={
            dataset:{"Quantum-assisted":values}
            for dataset,values in dataset_results.items()
        }

    rows=[]
    labels=[]

    for dataset,methods in source.items():
        for method,values in methods.items():
            rows.append(np.mean(np.asarray(values,dtype=float),axis=0))
            labels.append(f"{dataset}\n{method}")

    data=np.asarray(rows,dtype=float)
    x=np.arange(len(labels))
    width=0.2

    plt.figure()

    plt.bar(x-width*1.5,data[:,0],width,label="Precision@5")
    plt.bar(x-width/2,data[:,1],width,label="Precision@10")
    plt.bar(x+width/2,data[:,2],width,label="MAP@10")
    plt.bar(x+width*1.5,data[:,3],width,label="NDCG@10")

    plt.xticks(x,labels,rotation=20,ha="right")
    plt.legend()
    plt.title("IR Metric Comparison by Ranking Method")
    display_current_plot()

    plt.figure()

    sns.heatmap(data,annot=True,cmap="viridis",
                xticklabels=["P@5","P@10","MAP@10","NDCG@10"],
                yticklabels=labels)

    display_current_plot()
    plt.figure()

    for label,row in zip(labels,data):
        plt.plot([5,10],[row[0],row[1]],marker="o",label=label.replace("\n"," - "))

    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title("Precision@K Curve")
    plt.legend()
    display_current_plot()


def report_menu_error(action,exc):

    print(f"{action} failed: {type(exc).__name__}: {exc}")


def read_input(prompt):

    """Read interactive input without crashing on closed terminals."""

    try:
        return input(prompt)
    except (EOFError,KeyboardInterrupt):
        print("\nInput closed; exiting.")
        return None


def _fit_pipeline(train_docs,train_labels,raw_embeddings,max_samples,topic_sets=None):

    global docs,labels,embeddings,feature_weights,kmeans_model,cluster_labels,index

    docs=list(train_docs)
    labels=list(train_labels)
    embeddings=np.asarray(raw_embeddings)
    _build_tfidf_index()
    clear_ranking_caches()

    feature_weights=analyze_features(embeddings)
    embeddings=apply_feature_weights(embeddings,feature_weights)
    kmeans_model,cluster_labels=cluster_documents(embeddings)
    index=build_faiss_index(embeddings)

    indices=sample_training_indices(labels,max_samples,seed=42)

    if topic_sets is None:
        train_qsvm(embeddings[indices],np.asarray(labels)[indices])
    else:
        train_qsvm(
            embeddings[indices],
            np.asarray(labels)[indices],
            topic_sets=[topic_sets[i] for i in indices],
        )

    train_quantum_relevance_ranker(
        embeddings,labels,docs,topic_sets=topic_sets,
    )


def _validation_indices(docs,labels):

    fit_indices,_,validation_indices,_=split_documents(
        list(range(len(docs))),labels,eval_fraction=0.2,seed=43
    )

    return (
        np.asarray(fit_indices,dtype=int),
        np.asarray(validation_indices,dtype=int),
    )


@_transactional_pipeline
def train_newsgroups_model():

    global current_dataset,docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,feature_weights,kmeans_model,cluster_labels,index
    global ranking_weights

    current_dataset="newsgroups"

    all_docs,all_labels=load_newsgroups(DATA_ROOT / "20_newsgroups")
    docs,labels,eval_docs,eval_labels=split_documents(all_docs,all_labels)
    full_train_docs=list(docs)
    full_train_labels=list(labels)
    doc_topics=None
    eval_topics=None
    print("Training documents:",len(docs),"Evaluation documents:",len(eval_docs))

    print("Loading embedding model and generating embeddings...",flush=True)
    raw_embeddings,model=generate_embeddings(
        docs,f"{current_dataset}_train",labels=labels
    )
    fit_indices,validation_indices=_validation_indices(docs,labels)
    fit_docs=[docs[i] for i in fit_indices]
    fit_labels=[labels[i] for i in fit_indices]
    validation_docs=[docs[i] for i in validation_indices]
    validation_labels=[labels[i] for i in validation_indices]

    print("Fitting validation pipeline and tuning ranking weights...",flush=True)
    _fit_pipeline(fit_docs,fit_labels,raw_embeddings[fit_indices],120)
    ranking_weights,objective,validation_query_count=tune_ranking_weights(
        validation_docs,validation_labels,fit_labels,n_per_class=1
    )
    print(
        "Validation tuning complete:",ranking_weights,
        "using",validation_query_count,"queries; objective",objective,
        flush=True,
    )

    print("Refitting final pipeline on all training documents...",flush=True)
    _fit_pipeline(full_train_docs,full_train_labels,raw_embeddings,120)
    docs=full_train_docs
    labels=full_train_labels
    save_model_state(current_dataset,docs,labels)
    print("Training complete. Saved pipeline:",model_state_path(current_dataset),flush=True)


@_transactional_pipeline
def train_reuters_model():

    global current_dataset,docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,feature_weights,kmeans_model,cluster_labels,index
    global ranking_weights

    current_dataset="reuters"

    docs,labels,doc_topics=load_reuters(
        DATA_ROOT / "reuters21578",split="TRAIN",return_topics=True
    )
    eval_docs,eval_labels,eval_topics=load_reuters(
        DATA_ROOT / "reuters21578",split="TEST",return_topics=True
    )
    (
        docs,labels,eval_docs,eval_labels,doc_topics,eval_topics
    )=remove_cross_split_duplicates(
        docs,labels,eval_docs,eval_labels,
        train_topics=doc_topics,eval_topics=eval_topics,
    )
    full_train_docs=list(docs)
    full_train_labels=list(labels)
    full_train_topics=list(doc_topics)
    print("Training documents:",len(docs),"Evaluation documents:",len(eval_docs))

    print("Loading embedding model and generating embeddings...",flush=True)
    raw_embeddings,model=generate_embeddings(
        docs,f"{current_dataset}_train",labels=labels
    )
    fit_indices,validation_indices=_validation_indices(docs,labels)
    fit_docs=[docs[i] for i in fit_indices]
    fit_labels=[labels[i] for i in fit_indices]
    fit_topics=[doc_topics[i] for i in fit_indices]
    validation_docs=[docs[i] for i in validation_indices]
    validation_labels=[labels[i] for i in validation_indices]
    validation_topics=[doc_topics[i] for i in validation_indices]

    print("Fitting validation pipeline and tuning ranking weights...",flush=True)
    _fit_pipeline(fit_docs,fit_labels,raw_embeddings[fit_indices],200,fit_topics)
    ranking_weights,objective,validation_query_count=tune_ranking_weights(
        validation_docs,validation_labels,fit_labels,
        query_topics=validation_topics,corpus_topics=fit_topics,n_per_class=1,
    )
    print(
        "Validation tuning complete:",ranking_weights,
        "using",validation_query_count,"queries; objective",objective,
        flush=True,
    )

    print("Refitting final pipeline on all training documents...",flush=True)
    _fit_pipeline(
        full_train_docs,full_train_labels,raw_embeddings,200,full_train_topics
    )
    docs=full_train_docs
    labels=full_train_labels
    doc_topics=full_train_topics
    save_model_state(current_dataset,docs,labels)
    print("Training complete. Saved pipeline:",model_state_path(current_dataset),flush=True)


@_transactional_pipeline
def rebuild_model_for_search(choice):

    global current_dataset,docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,feature_weights,kmeans_model,cluster_labels,index

    if choice=="1":
        current_dataset="newsgroups"
        all_docs,all_labels=load_newsgroups(DATA_ROOT / "20_newsgroups")
        docs,labels,eval_docs,eval_labels=split_documents(all_docs,all_labels)
        doc_topics=None
        eval_topics=None
    elif choice=="2":
        current_dataset="reuters"
        docs,labels,doc_topics=load_reuters(
            DATA_ROOT / "reuters21578",split="TRAIN",return_topics=True
        )
        eval_docs,eval_labels,eval_topics=load_reuters(
            DATA_ROOT / "reuters21578",split="TEST",return_topics=True
        )
        (
            docs,labels,eval_docs,eval_labels,doc_topics,eval_topics
        )=remove_cross_split_duplicates(
            docs,labels,eval_docs,eval_labels,
            train_topics=doc_topics,eval_topics=eval_topics,
        )
    else:
        raise ValueError("Model choice must be 1 or 2")

    embeddings,model=generate_embeddings(
        docs,f"{current_dataset}_train",labels=labels
    )
    restore_model_state(current_dataset,docs,labels,embeddings)
    print("Trained pipeline loaded.")


def menu():

    global docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,index,current_dataset
    global feature_weights,kmeans_model,cluster_labels

    while True:

        print("\nQuantum Information Ranking System")

        print("1 Train Model on 20 Newsgroups")
        print("2 Train Model on Reuters")
        print("3 Choose Model for Search")
        print("4 Search Documents")
        print("5 Run Evaluation on held-out documents (Precision/MAP/NDCG)")
        print("6 Show Evaluation Graphs")
        print("7 Run Multi-seed Evaluation (mean and variation)")
        print("8 Exit")

        choice=read_input("Enter choice: ")

        if choice is None:
            break

        if choice=="1":
            try:
                train_newsgroups_model()
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Newsgroups training",exc)

        elif choice=="2":
            try:
                train_reuters_model()
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Reuters training",exc)

        elif choice=="3":

            print("1 Newsgroups")
            print("2 Reuters")

            c=read_input("> ")

            if c is None:
                break

            try:
                rebuild_model_for_search(c)
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Model selection",exc)

        elif choice=="4":
            if model is None or index is None:
                print("Train or load a model first.")
                continue
            query=read_input("Enter search query: ")

            if query is None:
                break

            try:
                ranked=hybrid_ranking(query)
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Search",exc)
                continue

            print("\nTop Results:\n")

            for rank,i in enumerate(ranked[:20],1):
                print(f"{rank}. Document {i}: {format_snippet(docs[i])}\n")

            # No Precision/MAP/NDCG here: for a free-text query typed by a
            # user, there's no independently known set of "correct" answers
            # to score against. Computing metrics from whatever the model
            # just retrieved would be circular (the model grading itself).
            # Use option 5 for metrics, which uses ground truth derived from
            # the dataset's own labels instead.
            print("(No ground truth exists for a free-text query, so no")
            print(" Precision/MAP/NDCG is shown here. Use option 5 to run")
            print(" evaluation against known relevant documents.)")

        elif choice=="5":
            if model is None or index is None:
                print("Train or load a model first.")
                continue
            try:
                evaluate_model()
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Evaluation",exc)

        elif choice=="6":
            try:
                show_metrics()
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Graph display",exc)

        elif choice=="7":
            try:
                evaluate_across_seeds()
            except (OSError,RuntimeError,ValueError,TypeError,ImportError,
                    pickle.UnpicklingError,EOFError) as exc:
                report_menu_error("Multi-seed evaluation",exc)

        elif choice=="8":
            break

        else:
            print("Invalid choice. Please choose a valid option (1-8).")


if __name__=="__main__":
    menu()
