"""Information-retrieval metrics and evaluation-query construction."""
import numpy as np

def normalize_scores(scores):

    """Map a score vector to [0, 1] without changing its ordering."""

    values=np.asarray(scores,dtype=float)

    if values.size==0:
        return values

    finite=np.isfinite(values)

    if not finite.any():
        return np.zeros_like(values)

    floor=np.min(values[finite])
    values=np.where(finite,values,floor)

    low=np.min(values)
    high=np.max(values)

    if high-low<=np.finfo(float).eps:
        return np.full_like(values,0.5)

    return (values-low)/(high-low)



def precision_at_k(ranked,relevant,k):

    """Precision at k, using the available result count for short rankings."""

    if k<=0:
        raise ValueError("k must be positive")

    available=min(k,len(ranked))

    if available==0:
        return 0.0

    rel=sum([1 for i in ranked[:available] if i in relevant])

    return rel/available



def compute_map(ranked,relevant,cutoff=10):

    if cutoff<=0:
        raise ValueError("cutoff must be positive")

    relevant=set(relevant)

    if len(relevant)==0:
        return 0.0

    score=0
    hits=0

    for i,doc in enumerate(ranked[:cutoff],start=1):

        if doc in relevant:
            hits+=1
            score+=hits/i

    if hits==0:
        return 0.0

    return score/min(len(relevant),cutoff)



def compute_ndcg(ranked,relevant,cutoff=10):

    if cutoff<=0:
        raise ValueError("cutoff must be positive")

    relevant=set(relevant)

    dcg=0

    for i,doc in enumerate(ranked[:cutoff]):

        rel=1 if doc in relevant else 0

        dcg+=rel/np.log2(i+2)

    ideal_hits=min(cutoff,len(relevant))
    idcg=sum([1/np.log2(i+2) for i in range(ideal_hits)])

    return dcg/idcg if idcg>0 else 0.0



def build_eval_queries(query_docs,query_labels,corpus_labels,
                       query_topics=None,corpus_topics=None,
                       n_per_class=2,query_words=15,min_class_size=2,seed=42):

    # Builds evaluation queries with relevance judgments that are known
    # BEFORE any retrieval happens, so the model can't grade its own homework.
    #
    # Query documents are held out from the indexed corpus. Their labels come
    # from the dataset, while relevant IDs refer only to indexed documents.

    rng=np.random.RandomState(seed)

    query_labels=np.asarray(query_labels)
    corpus_labels=np.asarray(corpus_labels)

    if (query_topics is None)!=(corpus_topics is None):
        raise ValueError("query_topics and corpus_topics must be provided together")

    if query_topics is not None and (
        len(query_topics)!=len(query_labels) or len(corpus_topics)!=len(corpus_labels)
    ):
        raise ValueError("Topic metadata must align with its documents")

    corpus_by_class={}
    query_by_class={}

    for idx,lab in enumerate(corpus_labels):
        corpus_by_class.setdefault(lab,[]).append(idx)

    for idx,lab in enumerate(query_labels):
        query_by_class.setdefault(lab,[]).append(idx)

    queries=[]

    for lab,query_indices in query_by_class.items():

        default_relevant_indices=corpus_by_class.get(lab,[])

        query_indices=np.array(query_indices)
        rng.shuffle(query_indices)

        chosen=query_indices[:min(n_per_class,len(query_indices))]

        for source_idx in chosen:

            words=query_docs[source_idx].split()

            if len(words)<3:
                continue

            query_text=" ".join(words[:query_words])

            if query_topics is None:
                relevant_indices=default_relevant_indices
            else:
                source_topics=set(query_topics[source_idx])
                relevant_indices=[
                    i for i,topics in enumerate(corpus_topics)
                    if source_topics.intersection(topics)
                ]

            if len(relevant_indices)<min_class_size:
                continue

            relevant=set(int(i) for i in relevant_indices)

            if len(relevant)==0:
                continue

            queries.append({
                "query":query_text,
                "source_idx":int(source_idx),
                "label":lab,
                "relevant":relevant,
            })

    return queries
