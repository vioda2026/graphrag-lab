# Experiment Matrix Template (M1)

| Exp ID | Structure Variant (`graph_builder`) | Exploration Variant (`graph_explorer`) | Retriever | Reader | Dataset Adapter | Metric | Notes |
|---|---|---|---|---|---|---|---|
| S1 | baseline | baseline | lexical | extractive | toy | avg exact-match-like | sanity baseline |
| S2 | **new structure only** | baseline | lexical | extractive | toy | avg metric | isolate graph effects |
| E1 | baseline | **new exploration only** | lexical | extractive | toy | avg metric | isolate traversal effects |
| J1 | **new structure** | **new exploration** | lexical | extractive | toy | avg metric | joint ablation |
