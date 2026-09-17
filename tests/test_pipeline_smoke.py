import unittest

import numpy as np

import main


class _DeterministicEmbeddingModel:
    """Small fixture model for exercising the pipeline without downloading a model."""

    def encode(self, texts):
        return np.tile(np.array([[1.0, 0.8, 0.2, 0.1]], dtype=float), (len(texts), 1))


class PipelineSmokeTests(unittest.TestCase):
    def test_small_pipeline_trains_and_ranks_a_query(self):
        previous_state = {name: getattr(main, name) for name in main._RUNTIME_STATE_NAMES}

        try:
            all_docs = [
                "Quantum kernels compare compact feature vectors.",
                "Quantum circuits can support relevance ranking.",
                "Statevector simulation is useful for small experiments.",
                "Classical retrieval uses semantic document embeddings.",
                "TF-IDF provides a lexical retrieval baseline.",
                "Information retrieval evaluates ranked documents.",
            ]
            all_labels = ["quantum", "quantum", "quantum", "retrieval", "retrieval", "retrieval"]
            train_docs, train_labels, _, _ = main.split_documents(
                all_docs,
                all_labels,
                eval_fraction=0.2,
                seed=42,
            )
            raw_embeddings = np.array(
                [
                    [1.0, 0.9, 0.1, 0.2],
                    [0.9, 1.0, 0.2, 0.1],
                    [0.8, 0.8, 0.2, 0.2],
                    [0.1, 0.2, 1.0, 0.9],
                ],
                dtype=float,
            )

            main._fit_pipeline(
                train_docs,
                train_labels,
                raw_embeddings,
                max_samples=len(train_docs),
            )
            main.model = _DeterministicEmbeddingModel()

            candidates, _ = main.retrieve_candidates("quantum ranking", k=3)
            ranked = main.hybrid_ranking("quantum ranking", candidates=candidates)

            self.assertGreaterEqual(len(candidates), 1)
            self.assertEqual(len(ranked), len(candidates))
            self.assertEqual(set(ranked.tolist()), set(candidates.tolist()))
        finally:
            for name, value in previous_state.items():
                setattr(main, name, value)
            main.clear_ranking_caches()


if __name__ == "__main__":
    unittest.main()
