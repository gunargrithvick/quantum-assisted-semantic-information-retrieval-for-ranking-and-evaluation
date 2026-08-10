"""Dataset loading and deterministic train/evaluation splitting for IR."""
import os
import numpy as np

from text import clean_text

def load_newsgroups(path,progress=True):

    path=os.fspath(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Newsgroups dataset directory not found: {path}")

    docs=[]
    labels=[]
    skipped=[]
    label_id=0
    processed=0

    with os.scandir(path) as entries:
        categories=sorted(
            [entry for entry in entries if entry.is_dir()],
            key=lambda entry:entry.name,
        )

    for category_entry in categories:

        category_path=category_entry.path

        with os.scandir(category_path) as entries:
            files=sorted(
                [entry for entry in entries if entry.is_file()],
                key=lambda entry:entry.name,
            )

        for file_entry in files:

            file_path=file_entry.path
            processed+=1

            try:

                with open(file_path,encoding="latin-1") as f:

                    text=f.read()

                    parts=text.split("\n\n",1)

                    if len(parts)>1:
                        text=parts[1]

                    docs.append(clean_text(text))
                    labels.append(label_id)

            except (OSError,UnicodeError) as exc:
                skipped.append((file_path,str(exc)))

            if progress and processed%1000==0:
                print(f"Newsgroups files processed: {processed}; documents loaded: {len(docs)}",flush=True)

        label_id+=1

    print("Documents loaded:",len(docs))

    if skipped:
        print("Files skipped:",len(skipped))
        for file_path,error in skipped[:5]:
            print(f"  {file_path}: {error}")

    return docs,labels



def load_reuters(path,split=None,return_topics=False,progress=True):

    path=os.fspath(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Reuters dataset directory not found: {path}")

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required for Reuters loading. "
            "Install the dependencies from requirements.txt."
        ) from exc

    docs=[]
    labels=[]
    topic_sets=[]

    with os.scandir(path) as entries:
        sgm_files=sorted(
            [entry.name for entry in entries
             if entry.is_file() and entry.name.endswith(".sgm")],
        )

    for file_index,file in enumerate(sgm_files,start=1):

        with open(os.path.join(path,file),encoding="latin-1") as f:

            soup=BeautifulSoup(f.read(),"html.parser")

            for article in soup.find_all("reuters"):

                article_split=(article.get("lewissplit") or "").upper()

                if split is not None and article_split != split.upper():
                    continue

                text_tag=article.find("text")

                if text_tag is None:
                    continue

                text=clean_text(text_tag.get_text())

                topics=article.find("topics")

                topic_names=tuple(sorted({
                    tag.get_text(strip=True).casefold()
                    for tag in topics.find_all("d")
                })) if topics else ()

                if topic_names:

                    primary_topic=topic_names[0]

                    docs.append(text)
                    labels.append(primary_topic)
                    topic_sets.append(topic_names)

        if progress:
            print(f"Reuters files processed: {file_index}/{len(sgm_files)}; documents loaded: {len(docs)}",flush=True)

    print("Documents loaded:",len(docs))

    if return_topics:
        return docs,labels,topic_sets

    return docs,labels



def split_documents(docs,labels,eval_fraction=0.2,seed=42):

    """Create a deterministic per-class train/evaluation split."""

    if len(docs)!=len(labels):
        raise ValueError("Documents and labels must have the same length")

    if not 0<=eval_fraction<1:
        raise ValueError("eval_fraction must be between 0 and 1")

    rng=np.random.RandomState(seed)
    labels=np.asarray(labels)
    train_indices=[]
    eval_indices=[]

    for lab in np.unique(labels):
        class_indices=np.flatnonzero(labels==lab)
        rng.shuffle(class_indices)

        if len(class_indices)<2:
            train_indices.extend(class_indices.tolist())
            continue

        n_eval=(
            0
            if eval_fraction==0
            else max(1,int(round(len(class_indices)*eval_fraction)))
        )
        n_eval=min(n_eval,len(class_indices)-1)
        eval_indices.extend(class_indices[:n_eval].tolist())
        train_indices.extend(class_indices[n_eval:].tolist())

    train_indices=np.asarray(sorted(train_indices),dtype=int)
    eval_indices=np.asarray(sorted(eval_indices),dtype=int)

    train_docs=[docs[i] for i in train_indices]
    train_labels=labels[train_indices].tolist()
    held_out_docs=[docs[i] for i in eval_indices]
    held_out_labels=labels[eval_indices].tolist()

    return train_docs,train_labels,held_out_docs,held_out_labels
