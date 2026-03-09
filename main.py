import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans

import faiss

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


docs=None
labels=None
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

dataset_results={}


def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z ]"," ",text)
    return text


def load_newsgroups(path):

    docs=[]
    labels=[]
    label_id=0

    for category in os.listdir(path):

        category_path=os.path.join(path,category)

        if os.path.isdir(category_path):

            for file in os.listdir(category_path):

                try:

                    with open(os.path.join(category_path,file),encoding="latin-1") as f:

                        text=f.read()

                        parts=text.split("\n\n",1)

                        if len(parts)>1:
                            text=parts[1]

                        docs.append(clean_text(text))
                        labels.append(label_id)

                except:
                    pass

            label_id+=1

    print("Documents loaded:",len(docs))

    return docs,labels


def load_reuters(path):

    from bs4 import BeautifulSoup

    docs=[]
    labels=[]
    label_map={}
    label_id=0

    for file in os.listdir(path):

        if file.endswith(".sgm"):

            with open(os.path.join(path,file),encoding="latin-1") as f:

                soup=BeautifulSoup(f.read(),"html.parser")

                for article in soup.find_all("reuters"):

                    text_tag=article.find("text")

                    if text_tag is None:
                        continue

                    text=clean_text(text_tag.get_text())

                    topics=article.find("topics")

                    if topics and topics.find("d"):

                        topic=topics.find("d").text

                        if topic not in label_map:

                            label_map[topic]=label_id
                            label_id+=1

                        docs.append(text)
                        labels.append(label_map[topic])

    print("Documents loaded:",len(docs))

    return docs,labels


def generate_embeddings(docs,dataset_name):

    cache_path=f"embeddings_{dataset_name}.npy"

    if os.path.exists(cache_path):

        embeddings=np.load(cache_path)

        model=SentenceTransformer("all-MiniLM-L6-v2")

        return embeddings,model

    device="cuda" if torch.cuda.is_available() else "cpu"

    model=SentenceTransformer("all-MiniLM-L6-v2",device=device)

    embeddings=model.encode(docs,batch_size=256,show_progress_bar=True)

    np.save(cache_path,embeddings)

    return embeddings,model


def analyze_features(X):

    scaler=StandardScaler()

    X_scaled=scaler.fit_transform(X)

    variances=np.var(X_scaled,axis=0)

    weights=variances/np.sum(variances)

    idx=np.argsort(variances)[::-1][:20]

    plt.figure()
    plt.bar(range(20),variances[idx])
    plt.title("Top Feature Importance")
    plt.show()
    plt.close()

    return weights


def apply_feature_weights(X,weights):

    if weights is None:
        return X

    return X*weights


def apply_pca(X):

    scaler=MinMaxScaler()

    X_scaled=scaler.fit_transform(X)

    pca=PCA(n_components=4)

    reduced=pca.fit_transform(X_scaled)

    return reduced,pca,scaler


def compute_quantum_kernel(X):

    feature_map=ZZFeatureMap(feature_dimension=X.shape[1],reps=2)

    kernel=FidelityQuantumKernel(feature_map=feature_map)

    K = kernel.evaluate(X.tolist(), X.tolist())

    return np.array(K)


def quantum_clustering(embeddings,n_clusters=10):

    global cluster_pca,cluster_scaler

    reduced,cluster_pca,cluster_scaler=apply_pca(embeddings)

    kmeans=KMeans(n_clusters=n_clusters,random_state=42)

    cluster_labels=kmeans.fit_predict(reduced)

    return kmeans,cluster_labels


def build_faiss_index(embeddings):

    dim=embeddings.shape[1]

    index=faiss.IndexFlatL2(dim)

    index.add(embeddings.astype("float32"))

    return index


def train_qsvm(embeddings,labels):

    global qsvm_model,qsvm_pca,qsvm_scaler,qsvm_kernel,qsvm_train_reduced

    reduced,qsvm_pca,qsvm_scaler=apply_pca(embeddings)
    qsvm_train_reduced=reduced

    feature_map=ZZFeatureMap(feature_dimension=reduced.shape[1],reps=2)
    qsvm_kernel=FidelityQuantumKernel(feature_map=feature_map)

    kernel_matrix = qsvm_kernel.evaluate(reduced.tolist(), reduced.tolist())

    qsvm_model=SVC(kernel="precomputed")

    qsvm_model.fit(kernel_matrix,labels)


def retrieve_candidates(query,k=25):

    global model,index,feature_weights
    global cluster_pca,cluster_scaler,kmeans_model,cluster_labels

    q_embed=model.encode([query])

    q_embed=apply_feature_weights(q_embed,feature_weights)

    distances,indices=index.search(q_embed.astype("float32"),k)

    candidates=indices[0]

    reduced_query=cluster_pca.transform(cluster_scaler.transform(q_embed))

    cluster_id = kmeans_model.predict(reduced_query)[0] if kmeans_model is not None and cluster_labels is not None else None

    cluster_docs=np.where(cluster_labels==cluster_id)[0] if cluster_id is not None and cluster_labels is not None else []

    cluster_set=set(cluster_docs)

    candidates=[i for i in candidates if i in cluster_set]

    if len(candidates)==0:
        candidates=indices[0]

    return np.array(candidates),q_embed


def hybrid_ranking(query):

    global embeddings, qsvm_pca, qsvm_scaler, qsvm_kernel, qsvm_model, qsvm_train_reduced

    candidates,q_embed=retrieve_candidates(query)

    candidate_embeddings=embeddings[candidates]

    classical_scores=cosine_similarity(q_embed,candidate_embeddings)[0]

    reduced_docs=qsvm_pca.transform(qsvm_scaler.transform(candidate_embeddings))
    reduced_query=qsvm_pca.transform(qsvm_scaler.transform(q_embed))

    kernel=qsvm_kernel if qsvm_kernel is not None and qsvm_train_reduced is not None else None

    if kernel is None:

        quantum_scores=np.zeros(len(candidate_embeddings))
        qsvm_scores=np.zeros(len(candidate_embeddings))

    else:

        quantum_matrix = kernel.evaluate(reduced_query.tolist(), reduced_docs.tolist())
        quantum_scores = np.array(quantum_matrix)[0]

        qsvm_scores = []

        for doc in reduced_docs:

            doc_kernel = qsvm_kernel.evaluate(
                [doc],
                qsvm_train_reduced.tolist()
            )

            score = qsvm_model.decision_function(doc_kernel).mean()

            qsvm_scores.append(score)

        qsvm_scores = np.array(qsvm_scores)

    final_scores=0.5*classical_scores+0.3*quantum_scores+0.2*qsvm_scores

    ranked=np.argsort(final_scores)[::-1]

    return candidates[ranked]


def precision_at_k(ranked,relevant,k):

    rel=sum([1 for i in ranked[:k] if i in relevant])

    return rel/k


def compute_map(ranked,relevant):

    score=0
    hits=0

    for i,doc in enumerate(ranked[:10]):

        if doc in relevant:
            hits+=1
            score+=hits/(i+1)

    if hits==0:
        return 0

    return score/hits


def compute_ndcg(ranked,relevant):

    dcg=0

    for i,doc in enumerate(ranked[:10]):

        rel=1 if doc in relevant else 0

        dcg+=rel/np.log2(i+2)

    idcg=sum([1/np.log2(i+2) for i in range(10)])

    return dcg/idcg


def show_metrics():

    global embeddings

    if embeddings is not None:
        analyze_features(embeddings)

    if len(dataset_results)==0:
        print("No evaluation results yet.")
        return

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
    plt.show()
    plt.close()

    data=np.array([[np.mean([x[i] for x in dataset_results[d]]) for i in range(4)] for d in names])

    plt.figure()

    sns.heatmap(data,annot=True,cmap="viridis",
                xticklabels=["P@5","P@10","MAP","NDCG"],
                yticklabels=names)

    plt.show()
    plt.close()
    plt.figure()

    for d in names:

        avg_p5=np.mean([x[0] for x in dataset_results[d]])
        avg_p10=np.mean([x[1] for x in dataset_results[d]])

        plt.plot([5,10],[avg_p5,avg_p10],marker="o",label=d)

    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title("Precision@K Curve")
    plt.legend()
    plt.show()
    plt.close()


def menu():

    global docs,labels,embeddings,model,index,current_dataset
    global feature_weights,kmeans_model,cluster_labels

    while True:

        print("\nQuantum Information Ranking System")

        print("1 Train Model on 20 Newsgroups")
        print("2 Train Model on Reuters")
        print("3 Choose Model for Search")
        print("4 Search Documents")
        print("5 Show Evaluation Graphs")
        print("6 Exit")

        choice=input("Enter choice: ")

        if choice=="1":

            current_dataset="newsgroups"

            docs,labels=load_newsgroups("dataset/20_newsgroups/20_newsgroups/20_newsgroups")

            embeddings,model=generate_embeddings(docs,current_dataset)

            feature_weights=analyze_features(embeddings)

            embeddings=apply_feature_weights(embeddings,feature_weights)

            kmeans_model,cluster_labels=quantum_clustering(embeddings)

            index=build_faiss_index(embeddings)

            indices=np.random.choice(len(labels),120,replace=False)
            train_qsvm(embeddings[indices],np.array(labels)[indices])

        elif choice=="2":

            current_dataset="reuters"

            docs,labels=load_reuters("dataset/reuters21578")

            embeddings,model=generate_embeddings(docs,current_dataset)

            feature_weights=analyze_features(embeddings)

            embeddings=apply_feature_weights(embeddings,feature_weights)

            kmeans_model,cluster_labels=quantum_clustering(embeddings)

            index=build_faiss_index(embeddings)

            indices=np.random.choice(len(labels),200,replace=False)
            train_qsvm(embeddings[indices],np.array(labels)[indices])

        elif choice=="3":

            print("1 Newsgroups")
            print("2 Reuters")

            c=input("> ")

            if c=="1":
                current_dataset="newsgroups"
                docs,labels=load_newsgroups("dataset/20_newsgroups/20_newsgroups/20_newsgroups")
            else:
                current_dataset="reuters"
                docs,labels=load_reuters("dataset/reuters21578")

            embeddings=np.load(f"embeddings_{current_dataset}.npy")

            model=SentenceTransformer("all-MiniLM-L6-v2")

            feature_weights=np.ones(embeddings.shape[1])

            embeddings=apply_feature_weights(embeddings,feature_weights)

            kmeans_model,cluster_labels=quantum_clustering(embeddings)

            index=build_faiss_index(embeddings)

            indices=np.random.choice(len(labels),120,replace=False)
            train_qsvm(embeddings[indices],np.array(labels)[indices])

            print("Model loaded.")

        elif choice=="4":
            if model is None or index is None:
                print("Train or load a model first.")
                continue
            query=input("Enter search query: ")

            ranked=hybrid_ranking(query)

            print("\nTop Results:\n")

            for rank,i in enumerate(ranked[:20],1):
                print(f"{rank}. {docs[i][:200]}\n")

            label_counts={}

            for i in ranked[:10]:
                label=labels[i]
                label_counts[label]=label_counts.get(label,0)+1

            query_label=max(label_counts,key=label_counts.get)

            relevant=set([i for i,l in enumerate(labels) if l==query_label])

            p5=precision_at_k(ranked,relevant,5)
            p10=precision_at_k(ranked,relevant,10)
            map_score=compute_map(ranked,relevant)
            ndcg_score=compute_ndcg(ranked,relevant)

            if current_dataset not in dataset_results:
                dataset_results[current_dataset]=[]

            dataset_results[current_dataset].append((p5,p10,map_score,ndcg_score))

            print("Precision@5:",p5)
            print("Precision@10:",p10)
            print("MAP:",map_score)
            print("NDCG:",ndcg_score)

        elif choice=="5":
            show_metrics()

        elif choice=="6":
            break

        else:
            print("Invalid choice. Please choose a valid option (1-6).")


if __name__=="__main__":
    menu()