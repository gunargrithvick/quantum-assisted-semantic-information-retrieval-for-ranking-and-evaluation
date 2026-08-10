"""Quantum-kernel ranking and feature-preparation utilities."""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from plotting import display_current_plot

def analyze_features(X,show_plot=False):

    """Return an unsupervised, training-only variance weighting heuristic.

    Each feature receives its variance divided by the mean feature variance,
    so the average weight remains one while higher-variance dimensions receive
    more emphasis. Training functions call this only on training embeddings.
    """

    X=np.asarray(X,dtype=float)

    if X.ndim!=2 or X.shape[0]==0 or X.shape[1]==0:
        raise ValueError("X must be a non-empty two-dimensional array")

    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")

    variances=np.var(X,axis=0)
    mean_variance=float(np.mean(variances))

    if mean_variance<=np.finfo(float).eps:
        weights=np.ones(X.shape[1],dtype=float)
    else:
        # Keep the average feature scale unchanged while emphasizing
        # dimensions that carry more variation in the original embeddings.
        weights=variances/mean_variance

    n_top=min(20,X.shape[1])
    idx=np.argsort(variances)[::-1][:n_top]

    plt.figure()
    plt.bar(range(n_top),variances[idx])
    plt.title("Top Feature Importance")

    if show_plot:
        display_current_plot()
    else:
        plt.close()

    return weights



def apply_feature_weights(X,weights):

    if weights is None:
        return X

    X=np.asarray(X)
    weights=np.asarray(weights)

    if X.ndim!=2 or weights.ndim!=1 or X.shape[1]!=weights.shape[0]:
        raise ValueError("Feature weights must match the embedding dimensions")

    if not np.isfinite(weights).all():
        raise ValueError("Feature weights must be finite")

    return X*weights



def normalize_embeddings(X):

    values=np.asarray(X,dtype=np.float32)

    if values.ndim!=2:
        raise ValueError("Embeddings must be a two-dimensional array")

    norms=np.linalg.norm(values,axis=1,keepdims=True)

    if np.any(norms<=np.finfo(np.float32).eps):
        raise ValueError("Embeddings must not contain zero vectors")

    return values/norms



def apply_pca(X):

    X=np.asarray(X)

    if X.ndim!=2 or X.shape[0]==0 or X.shape[1]==0:
        raise ValueError("PCA requires a non-empty two-dimensional array")

    scaler=MinMaxScaler()

    X_scaled=scaler.fit_transform(X)

    n_components=min(4,X_scaled.shape[0],X_scaled.shape[1])
    pca=PCA(n_components=n_components)

    reduced=pca.fit_transform(X_scaled)

    return reduced,pca,scaler



def _build_quantum_kernel(feature_dimension):

    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    except ImportError as exc:
        raise RuntimeError(
            "qiskit and qiskit-machine-learning are required for quantum ranking. "
            "Install the dependencies from requirements.txt."
        ) from exc

    feature_map=ZZFeatureMap(feature_dimension=feature_dimension,reps=2)
    return FidelityStatevectorKernel(
        feature_map=feature_map,
        shots=None,
        enforce_psd=True,
    )



def build_faiss_index(embeddings):

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is required for document search. "
            "Install the dependencies from requirements.txt."
        ) from exc

    dim=embeddings.shape[1]

    index=faiss.IndexFlatIP(dim)

    index.add(normalize_embeddings(embeddings))

    return index



def sample_training_indices(labels,max_samples,seed=42):

    labels=np.asarray(labels)

    if max_samples<=0:
        raise ValueError("max_samples must be positive")

    unique_labels=np.unique(labels)

    if len(unique_labels)<2:
        raise ValueError("QSVM training requires at least two classes")

    sample_size=min(max_samples,len(labels))

    if sample_size<2:
        raise ValueError("QSVM training requires at least two samples")

    rng=np.random.RandomState(seed)
    selected=[]

    class_order=np.array(unique_labels,copy=True)
    rng.shuffle(class_order)

    for lab in class_order[:min(sample_size,len(class_order))]:
        class_indices=np.flatnonzero(labels==lab)
        selected.append(int(rng.choice(class_indices)))

    selected_set=set(selected)
    remaining=np.array([i for i in range(len(labels)) if i not in selected_set])
    rng.shuffle(remaining)
    selected.extend(remaining[:sample_size-len(selected)].tolist())

    return np.asarray(selected,dtype=int)
