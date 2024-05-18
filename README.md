# Thesis
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/gh-mathjax@2.0.0"></script>

Bachelor thesis on Random Localised Robust Weighted (RLRW) Conformal Prediction on Nonexchangeable Data \\

NOTE: Most methods will be translated from R to Python from https://github.com/rohanhore/RLCP

A large issue when creating confidence intervals relies on the assumption that the data in exchangeable, i.e. for any symmetric algorithm 
$$
\mathcal{A}(\{(X_i, Y_i)\}_{i \in [n]}) = \mathcal{N}(\{(X_\{\sigma(i), Y_{\sigma(i)}})\}_{i\in [n]})
$$

