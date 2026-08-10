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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler,MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans


from config import (
    CLASSICAL_RANK_WEIGHT,
    DATA_ROOT,
    EMBEDDING_CACHE_VERSION,
    EMBEDDING_MODEL_NAME,
    MODEL_STATE_VERSION,
    PROJECT_ROOT,
    QSVM_RANK_WEIGHT,
    QUANTUM_RANK_WEIGHT,
)
from data import load_newsgroups,load_reuters,split_documents
from embeddings import _input_fingerprint,generate_embeddings
from metrics import (
    build_eval_queries,
    compute_map,
    compute_ndcg,
    normalize_scores,
    precision_at_k,
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

dataset_results={}


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


def model_state_path(dataset_name):

    return PROJECT_ROOT / f"model_{dataset_name}_train.pkl"


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
        qsvm_train_reduced is not None
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
    }

    temporary_path=state_path.with_suffix(".tmp")
    temporary_path.write_bytes(pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL))
    temporary_path.replace(state_path)


def restore_model_state(dataset_name,docs,labels,raw_embeddings):

    global embeddings,feature_weights
    global cluster_pca,cluster_scaler,kmeans_model,cluster_labels
    global qsvm_model,qsvm_pca,qsvm_scaler,qsvm_kernel,qsvm_train_reduced,index
    global qsvm_label_binarizer,qsvm_multilabel

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

    if len(cluster_labels)!=len(embeddings):
        raise RuntimeError("Saved cluster labels do not match embeddings")

    qsvm_kernel=_build_quantum_kernel(qsvm_train_reduced.shape[1])
    index=build_faiss_index(embeddings)


def retrieve_candidates(query,k=25):

    global model,index,feature_weights
    global cluster_pca,cluster_scaler,kmeans_model,cluster_labels

    cleaned_query=clean_text(query)

    if not cleaned_query:
        raise ValueError("Search query must contain at least one non-whitespace character")

    q_embed=model.encode([cleaned_query])

    q_embed=apply_feature_weights(q_embed,feature_weights)

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

    if cluster_ready:
        reduced_query=cluster_pca.transform(cluster_scaler.transform(q_embed))
        cluster_id=kmeans_model.predict(reduced_query)[0]
        cluster_docs=np.flatnonzero(cluster_labels==cluster_id)
    else:
        cluster_docs=None

    if cluster_docs is None or len(cluster_docs)==0:
        search_k=min(k,index.ntotal)
        distances,indices=index.search(search_vector,search_k)
        candidates=indices[0]
    else:
        # IndexFlat search is already exhaustive. Request all results so a
        # relevant member outside the global top-k is not discarded before
        # cluster filtering.
        distances,indices=index.search(search_vector,index.ntotal)
        all_candidates=indices[0]
        candidates=all_candidates[np.isin(all_candidates,cluster_docs)][:k]

    candidates=candidates[candidates>=0]

    return np.array(candidates),q_embed


def hybrid_ranking(query):

    global embeddings, qsvm_pca, qsvm_scaler, qsvm_kernel, qsvm_model, qsvm_train_reduced
    global qsvm_multilabel

    candidates,q_embed=retrieve_candidates(query)

    if len(candidates)==0:
        return candidates

    candidate_embeddings=embeddings[candidates]

    classical_scores=normalize_scores(
        cosine_similarity(q_embed,candidate_embeddings)[0]
    )

    quantum_state_ready=(
        qsvm_pca is not None and
        qsvm_scaler is not None and
        qsvm_kernel is not None and
        qsvm_train_reduced is not None
    )

    if not quantum_state_ready:
        raise RuntimeError(
            "Quantum ranking state is unavailable. Train or load a quantum pipeline first."
        )

    reduced_docs=qsvm_pca.transform(qsvm_scaler.transform(candidate_embeddings))
    reduced_query=qsvm_pca.transform(qsvm_scaler.transform(q_embed))

    quantum_matrix=qsvm_kernel.evaluate(reduced_query.tolist(),reduced_docs.tolist())
    quantum_scores=normalize_scores(np.asarray(quantum_matrix)[0])

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

    final_scores=(
        CLASSICAL_RANK_WEIGHT*classical_scores+
        QUANTUM_RANK_WEIGHT*quantum_scores+
        QSVM_RANK_WEIGHT*qsvm_scores
    )

    ranked=np.argsort(final_scores)[::-1]

    return candidates[ranked]


def evaluate_model(n_per_class=2,verbose=True,reveal_queries=False):

    # Runs Precision@5, Precision@10, MAP, and NDCG against the ground truth
    # from build_eval_queries. Relevance here is decided by the dataset's
    # own category labels, independently of what the model retrieves â€”
    # unlike the old approach, which inferred "relevant" from the model's
    # own top results and then evaluated the model against that.

    global docs,labels,doc_topics,eval_docs,eval_labels,eval_topics,current_dataset

    if docs is None or labels is None or eval_docs is None or eval_labels is None:
        print("Train or load a model first.")
        return

    dataset_results.pop(current_dataset,None)

    eval_queries=build_eval_queries(
        eval_docs,eval_labels,labels,
        query_topics=eval_topics,corpus_topics=doc_topics,
        n_per_class=n_per_class
    )

    if len(eval_queries)==0:
        print("Could not build any evaluation queries (classes too small).")
        return

    results=[]

    for q in eval_queries:

        ranked=hybrid_ranking(q["query"])

        relevant=q["relevant"]

        p5=precision_at_k(ranked,relevant,5)
        p10=precision_at_k(ranked,relevant,10)
        map_score=compute_map(ranked,relevant)
        ndcg_score=compute_ndcg(ranked,relevant)

        results.append((p5,p10,map_score,ndcg_score))

        if verbose:
            query_display=(
                f" \"{format_snippet(q['query'],60)}\""
                if reveal_queries else ""
            )
            print(f"Query source {q['source_idx']} (label {q['label']}){query_display}")
            print(f"  P@5={p5:.3f}  P@10={p10:.3f}  MAP={map_score:.3f}  NDCG={ndcg_score:.3f}")

    results=np.asarray(results,dtype=float)
    dataset_results[current_dataset]=results.tolist()

    print("\n--- Evaluation summary (held-out queries,",len(eval_queries),"queries) ---")
    print("Precision@5 :",results[:,0].mean())
    print("Precision@10:",results[:,1].mean())
    print("MAP         :",results[:,2].mean())
    print("NDCG        :",results[:,3].mean())


def show_metrics():

    global embeddings

    if len(dataset_results)==0:
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

    names=list(dataset_results.keys())

    p5=[np.mean([x[0] for x in dataset_results[d]]) for d in names]
    p10=[np.mean([x[1] for x in dataset_results[d]]) for d in names]
    maps=[np.mean([x[2] for x in dataset_results[d]]) for d in names]
    ndcg=[np.mean([x[3] for x in dataset_results[d]]) for d in names]

    x=np.arange(len(names))
    width=0.2

    plt.figure()

    plt.bar(x-width*1.5,p5,width,label="Precision@5")
    plt.bar(x-width/2,p10,width,label="Precision@10")
    plt.bar(x+width/2,maps,width,label="MAP")
    plt.bar(x+width*1.5,ndcg,width,label="NDCG")

    plt.xticks(x,names)
    plt.legend()
    plt.title("IR Metric Comparison")
    display_current_plot()

    data=np.array([[np.mean([x[i] for x in dataset_results[d]]) for i in range(4)] for d in names])

    plt.figure()

    sns.heatmap(data,annot=True,cmap="viridis",
                xticklabels=["P@5","P@10","MAP","NDCG"],
                yticklabels=names)

    display_current_plot()
    plt.figure()

    for d in names:

        avg_p5=np.mean([x[0] for x in dataset_results[d]])
        avg_p10=np.mean([x[1] for x in dataset_results[d]])

        plt.plot([5,10],[avg_p5,avg_p10],marker="o",label=d)

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


@_transactional_pipeline
def train_newsgroups_model():

    global current_dataset,docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,feature_weights,kmeans_model,cluster_labels,index

    current_dataset="newsgroups"

    all_docs,all_labels=load_newsgroups(DATA_ROOT / "20_newsgroups")
    docs,labels,eval_docs,eval_labels=split_documents(all_docs,all_labels)
    doc_topics=None
    eval_topics=None
    print("Training documents:",len(docs),"Evaluation documents:",len(eval_docs))

    embeddings,model=generate_embeddings(
        docs,f"{current_dataset}_train",labels=labels
    )
    feature_weights=analyze_features(embeddings)
    embeddings=apply_feature_weights(embeddings,feature_weights)
    kmeans_model,cluster_labels=cluster_documents(embeddings)
    index=build_faiss_index(embeddings)
    indices=sample_training_indices(labels,120,seed=42)
    train_qsvm(embeddings[indices],np.asarray(labels)[indices])
    save_model_state(current_dataset,docs,labels)


@_transactional_pipeline
def train_reuters_model():

    global current_dataset,docs,labels,doc_topics,eval_docs,eval_labels,eval_topics
    global embeddings,model,feature_weights,kmeans_model,cluster_labels,index

    current_dataset="reuters"

    docs,labels,doc_topics=load_reuters(
        DATA_ROOT / "reuters21578",split="TRAIN",return_topics=True
    )
    eval_docs,eval_labels,eval_topics=load_reuters(
        DATA_ROOT / "reuters21578",split="TEST",return_topics=True
    )
    print("Training documents:",len(docs),"Evaluation documents:",len(eval_docs))

    embeddings,model=generate_embeddings(
        docs,f"{current_dataset}_train",labels=labels
    )
    feature_weights=analyze_features(embeddings)
    embeddings=apply_feature_weights(embeddings,feature_weights)
    kmeans_model,cluster_labels=cluster_documents(embeddings)
    index=build_faiss_index(embeddings)
    indices=sample_training_indices(labels,200,seed=42)
    train_qsvm(
        embeddings[indices],
        np.asarray(labels)[indices],
        topic_sets=[doc_topics[i] for i in indices],
    )
    save_model_state(current_dataset,docs,labels)


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
        print("7 Exit")

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
            break

        else:
            print("Invalid choice. Please choose a valid option (1-7).")


if __name__=="__main__":
    menu()
