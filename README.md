image retrieval experiments.

embedding: DINOv2


global feature:
- cosine similarity

local features:
- pairwise consine
- foregrond extration using pca
- and then aggregate by either:
    - hausdorff
    - mean hausdorff
    - unbalanced OT using sinkhorn's iteration
