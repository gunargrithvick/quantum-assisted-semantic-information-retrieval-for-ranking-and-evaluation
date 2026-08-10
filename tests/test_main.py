import unittest
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main


class MainFunctionTests(unittest.TestCase):

    def test_clean_text_is_consistent(self):
        self.assertEqual(main.clean_text("  Hello, WORLD! 123 "), "hello world 123")
        self.assertEqual(main.clean_text("Hello, WORLD!"), main.clean_text("hello world"))
        self.assertEqual(main.clean_text("C++ RFC-822 x86 .NET"), "c++ rfc-822 x86 .net")

    def test_format_snippet_is_bounded_and_single_line(self):
        self.assertEqual(main.format_snippet("one\n two",20),"one two")
        self.assertEqual(main.format_snippet("abcdefgh",6),"abc...")
        with self.assertRaises(ValueError):
            main.format_snippet("text",0)

    def test_model_state_loader_blocks_executable_globals(self):
        class Exploit:
            def __reduce__(self):
                return (eval,("1 + 1",))

        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.pkl"
            path.write_bytes(pickle.dumps(Exploit()))

            with self.assertRaises(pickle.UnpicklingError):
                main.load_model_state(path)

    def test_model_state_loader_accepts_pipeline_scalers(self):
        scaler=main.MinMaxScaler().fit(np.array([[0.0,1.0],[2.0,3.0]]))

        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.pkl"
            path.write_bytes(pickle.dumps(scaler))
            restored=main.load_model_state(path)

        self.assertIsInstance(restored,main.MinMaxScaler)

    def test_model_state_loader_accepts_qsvm_classifier_state(self):
        classifier=main.OneVsRestClassifier(
            main.SVC(kernel="precomputed")
        ).fit(
            np.eye(3),
            [[1,0],[0,1],[1,0]],
        )

        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.pkl"
            path.write_bytes(pickle.dumps(classifier))
            restored=main.load_model_state(path)

        self.assertIsInstance(restored,main.OneVsRestClassifier)

    def test_pipeline_switch_rolls_back_after_failure(self):
        previous={
            name:getattr(main,name)
            for name in main._RUNTIME_STATE_NAMES
        }

        main.current_dataset="reuters"
        main.docs=["old document"]
        main.labels=["old label"]
        main.model="old model"
        main.index="old index"

        with patch.object(main._app,"load_newsgroups",return_value=(["a","b"],[0,0])), \
             patch.object(main._app,"generate_embeddings",return_value=(np.ones((2,2)),"new model")), \
             patch.object(main._app,"restore_model_state",side_effect=RuntimeError("broken state")):
            with self.assertRaises(RuntimeError):
                main.rebuild_model_for_search("1")

        self.assertEqual(main.current_dataset,"reuters")
        self.assertEqual(main.docs,["old document"])
        self.assertEqual(main.labels,["old label"])
        self.assertEqual(main.model,"old model")
        self.assertEqual(main.index,"old index")

        for name,value in previous.items():
            setattr(main,name,value)

    def test_read_input_handles_closed_terminal(self):
        with patch("builtins.input",side_effect=EOFError):
            self.assertIsNone(main.read_input("> "))

    def test_legacy_star_import_exports_public_api(self):
        namespace={}
        exec("from main import *",namespace)
        self.assertEqual(namespace["clean_text"]("C++"),"c++")

    def test_menu_handles_missing_reuters_dependency(self):
        with patch.object(
            main._app,
            "train_reuters_model",
            side_effect=ModuleNotFoundError("No module named bs4"),
        ), patch.object(main._app,"read_input",side_effect=["2","8"]):
            main.menu()

    def test_menu_handles_corrupt_model_state(self):
        with patch.object(
            main._app,
            "rebuild_model_for_search",
            side_effect=pickle.UnpicklingError("invalid state"),
        ), patch.object(main._app,"read_input",side_effect=["3","1","8"]):
            main.menu()

    def test_split_documents_is_deterministic_and_disjoint(self):
        docs=[f"document {i} text" for i in range(20)]
        labels=[0]*10+[1]*10

        first=main.split_documents(docs,labels,eval_fraction=0.2,seed=42)
        second=main.split_documents(docs,labels,eval_fraction=0.2,seed=42)

        self.assertEqual(first,second)
        train_docs,train_labels,eval_docs,eval_labels=first
        self.assertEqual(len(train_docs),16)
        self.assertEqual(len(eval_docs),4)
        self.assertTrue(set(train_docs).isdisjoint(eval_docs))
        self.assertEqual(len(train_labels),len(train_docs))
        self.assertEqual(len(eval_labels),len(eval_docs))

        with self.assertRaises(ValueError):
            main.split_documents(["one"],[0,1])
        with self.assertRaises(ValueError):
            main.split_documents(["one"],[0],eval_fraction=1)

        no_eval=main.split_documents(["one","two"],[0,0],eval_fraction=0)
        self.assertEqual(len(no_eval[2]),0)

    def test_split_documents_removes_duplicate_text_before_splitting(self):
        train_docs,train_labels,eval_docs,eval_labels=main.split_documents(
            ["duplicate","duplicate","unique one","unique two"],
            [0,0,1,1],
            eval_fraction=0.5,
            seed=42,
        )

        self.assertTrue(set(train_docs).isdisjoint(eval_docs))
        self.assertEqual(len(train_docs)+len(eval_docs),3)
        self.assertEqual(len(train_labels),len(train_docs))
        self.assertEqual(len(eval_labels),len(eval_docs))

    def test_cross_split_duplicate_removal_filters_only_evaluation_records(self):
        result=main.remove_cross_split_duplicates(
            ["train text","shared text"],
            ["a","b"],
            ["shared text","held out"],
            ["b","c"],
        )

        self.assertEqual(result[0],["train text","shared text"])
        self.assertEqual(result[2],["held out"])
        self.assertEqual(result[3],["c"])

    def test_multilabel_relevance_uses_topic_overlap(self):
        queries=main.build_eval_queries(
            ["query alpha beta"],
            ["primary"],
            ["oil","grain"],
            query_topics=[("cocoa","trade")],
            corpus_topics=[("oil",),("grain","trade")],
            n_per_class=1,
            min_class_size=1,
        )

        self.assertEqual(len(queries),1)
        self.assertEqual(queries[0]["relevant"],{1})

    def test_metrics_have_bounded_denominators(self):
        self.assertAlmostEqual(main.precision_at_k([1,9],{1},2),0.5)
        self.assertAlmostEqual(main.precision_at_k([1,9],{1},5),0.5)
        self.assertAlmostEqual(main.recall_at_k([1,9],{1,2},2),0.5)
        self.assertEqual(main.precision_at_k([],{1},5),0.0)
        self.assertAlmostEqual(main.compute_map([1,9,8],{1,2}),0.5)
        self.assertAlmostEqual(main.compute_ndcg([1,9,8],{1}),1.0)
        self.assertEqual(main.compute_map([],set()),0.0)
        self.assertEqual(main.compute_ndcg([],set()),0.0)

    def test_map_and_ndcg_match_known_cutoff_examples(self):
        self.assertAlmostEqual(
            main.compute_map([3,1,2],{1,2},cutoff=3),
            (0.5+(2/3))/2,
        )

        expected_dcg=(1/np.log2(3))+(1/np.log2(4))
        ideal_dcg=1+(1/np.log2(3))
        self.assertAlmostEqual(
            main.compute_ndcg([3,1,2],{1,2},cutoff=3),
            expected_dcg/ideal_dcg,
        )

    def test_lexical_baseline_is_deterministic(self):
        previous_docs=main.docs
        try:
            main.docs=[
                "quantum kernel ranking",
                "classical document retrieval",
                "quantum document search",
            ]
            ranked=main.lexical_baseline_ranking("quantum",k=2)
            np.testing.assert_array_equal(ranked,np.array([0,2]))
        finally:
            main.docs=previous_docs

    def test_query_document_features_are_compact_and_finite(self):
        query=np.array([[1.0,0.0,0.0,0.0]])
        documents=np.array([
            [1.0,0.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
        ])

        features=main.build_query_document_features(query,documents)

        self.assertEqual(features.shape,(2,4))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[0,0],features[1,0])

    def test_tfidf_baseline_is_deterministic(self):
        previous_docs=main.docs
        previous_vectorizer=main.tfidf_vectorizer
        previous_matrix=main.tfidf_matrix
        try:
            main.docs=[
                "quantum kernel ranking",
                "classical document retrieval",
                "quantum document search",
            ]
            main._build_tfidf_index()
            ranked=main.tfidf_baseline_ranking("quantum",k=2)
            np.testing.assert_array_equal(ranked,np.array([2,0]))
        finally:
            main.docs=previous_docs
            main.tfidf_vectorizer=previous_vectorizer
            main.tfidf_matrix=previous_matrix

    def test_score_and_embedding_normalization(self):
        np.testing.assert_allclose(
            main.normalize_scores(np.array([-2.0,0.0,8.0])),
            np.array([0.0,0.2,1.0]),
        )
        normalized=main.normalize_embeddings(np.array([[3.0,4.0],[1.0,0.0]]))
        np.testing.assert_allclose(np.linalg.norm(normalized,axis=1),np.ones(2))

        with self.assertRaises(ValueError):
            main.apply_feature_weights(np.ones((2,3)),np.ones(2))

    def test_quantum_ranking_weights_are_dominant_and_normalized(self):
        self.assertAlmostEqual(
            main.CLASSICAL_RANK_WEIGHT+
            main.QUANTUM_RANK_WEIGHT+
            main.QSVM_RANK_WEIGHT,
            1.0,
        )
        self.assertGreater(
            main.QUANTUM_RANK_WEIGHT+main.QSVM_RANK_WEIGHT,
            main.CLASSICAL_RANK_WEIGHT,
        )

    def test_sampling_is_seeded_and_covers_classes(self):
        labels=["a","a","b","b","c","c"]
        first=main.sample_training_indices(labels,4,seed=7)
        second=main.sample_training_indices(labels,4,seed=7)

        np.testing.assert_array_equal(first,second)
        self.assertGreaterEqual(len(set(np.asarray(labels)[first])),2)

    def test_small_pca_and_clustering_dimensions(self):
        data=np.arange(6.0).reshape(3,2)
        reduced,_,_=main.apply_pca(data)
        self.assertEqual(reduced.shape,(3,2))
        _,cluster_labels=main.cluster_documents(data,n_clusters=10)
        self.assertEqual(len(cluster_labels),3)

    def test_missing_dataset_path_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            main.load_newsgroups(main.DATA_ROOT/"does-not-exist")

        with self.assertRaises(ValueError):
            main.generate_embeddings([],"empty")


if __name__=="__main__":
    unittest.main()
