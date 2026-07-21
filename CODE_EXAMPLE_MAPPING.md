# Code Example Mapping

Maps every `demo_*` function in `computations/` to its numerical-example
admonition in the online course companion (`docs/`). Regenerated 2026-07-21
after the Phase 2 port, the Phase 3 computation verification, and the Phase 5
Codex review (which rewrote `demo_permutation_importance` and
`demo_bias_variance_tradeoff` and hardened `verify_all.py`).

## Summary

- **17 computation files** containing **98 demo functions**
- **19 companion pages** (`docs/`: 10 modules + 8 deep-dive appendices + Q&A)
- Every function is referenced by exactly one `!!! example` admonition whose
  `*Source:*` line cites it. Verification (`pixi run -e compute verify`): **97/98**
  numerically verified (78 exact PASS + 19 documented prose/formatting
  adjudications), 1 qualitative example with no numbers, 0 unexplained flags,
  0 errors.

---

## Per-File Mapping

### module2_examples.py (5 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_gradient_descent_convergence` | docs/modules/02-regression.md | Numerical Example: Watching Gradient Descent Converge | 179 |
| `demo_learning_rate_effects` | docs/modules/02-regression.md | Numerical Example: Learning Rate Effects | 236 |
| `demo_vif_multicollinearity` | docs/modules/02-regression.md | Numerical Example: VIF Multicollinearity Detection | 466 |
| `demo_lasso_feature_selection` | docs/modules/02-regression.md | Numerical Example: Lasso Feature Selection | 541 |
| `demo_ridge_vs_lasso` | docs/modules/02-regression.md | Numerical Example: Ridge vs Lasso Comparison | 610 |

### module3_examples.py (6 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_sigmoid_function` | docs/modules/03-classification.md | Numerical Example: Sigmoid Function in Action | 66 |
| `demo_cost_based_threshold` | docs/modules/03-classification.md | Numerical Example: Cost-Based Threshold Selection | 204 |
| `demo_roc_curve_construction` | docs/modules/03-classification.md | Numerical Example: Building an ROC Curve Step by Step | 261 |
| `demo_gini_impurity_calculation` | docs/modules/03-classification.md | Numerical Example: Evaluating a Split with Gini | 456 |
| `demo_tree_overfitting` | docs/modules/03-classification.md | Numerical Example: Decision Tree Overfitting | 525 |
| `demo_feature_importance` | docs/modules/03-classification.md | Numerical Example: Feature Importance | 615 |

### module4_examples.py (7 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_ensemble_variance_correlation` | docs/modules/04-ensemble-methods.md | Numerical Example: Ensemble Variance and Correlation | 40 |
| `demo_bootstrap_sampling` | docs/modules/04-ensemble-methods.md | Numerical Example: Bootstrap Sampling in Action | 142 |
| `demo_rf_vs_single_tree` | docs/modules/04-ensemble-methods.md | Numerical Example: Random Forest vs Single Tree | 213 |
| `demo_oob_vs_cv` | docs/modules/04-ensemble-methods.md | Numerical Example: OOB Error vs Cross-Validation | 310 |
| `demo_gradient_boosting_steps` | docs/modules/04-ensemble-methods.md | Numerical Example: Gradient Boosting Step by Step | 433 |
| `demo_learning_rate_effects` | docs/modules/04-ensemble-methods.md | Numerical Example: Learning Rate Effects on Boosting | 494 |
| `demo_early_stopping` | docs/modules/04-ensemble-methods.md | Numerical Example: Early Stopping in Action | 585 |

### module5_examples.py (8 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_kmeans_iterations` | docs/modules/05-unsupervised.md | Numerical Example: K-Means Iterations Step by Step | 124 |
| `demo_elbow_silhouette` | docs/modules/05-unsupervised.md | Numerical Example: Elbow and Silhouette Comparison | 222 |
| `demo_silhouette_by_hand` | docs/modules/05-unsupervised.md | Numerical Example: Silhouette Score by Hand | 262 |
| `demo_dbscan_classification` | docs/modules/05-unsupervised.md | Numerical Example: DBSCAN Core, Border, and Noise | 344 |
| `demo_pca_variance_explained` | docs/modules/05-unsupervised.md | Numerical Example: PCA Variance Explained | 505 |
| `demo_pca_loadings` | docs/modules/05-unsupervised.md | Numerical Example: PCA Loadings Interpretation | 576 |
| `demo_tsne_perplexity` | docs/modules/05-unsupervised.md | Numerical Example: t-SNE Perplexity Sensitivity | 665 |
| `demo_pca_vs_tsne` | docs/modules/05-unsupervised.md | Numerical Example: PCA vs t-SNE on Structured Data | 760 |

### module6_examples.py (8 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_parameter_counting` | docs/modules/06-neural-networks.md | Numerical Example: Parameter Counting Walkthrough | 308 |
| `demo_relu_vs_sigmoid_gradients` | docs/modules/06-neural-networks.md | Numerical Example: ReLU vs Sigmoid Gradients | 205 |
| `demo_xor_hidden_layer` | docs/modules/06-neural-networks.md | Numerical Example: XOR with a Hidden Layer | 77 |
| `demo_cross_entropy_vs_mse` | docs/modules/06-neural-networks.md | Numerical Example: Cross-Entropy vs MSE for Classification | 405 |
| `demo_backprop_by_hand` | docs/modules/06-neural-networks.md | Numerical Example: Backpropagation by Hand | 460 |
| `demo_learning_rate_effects_nn` | docs/modules/06-neural-networks.md | Numerical Example: Learning Rate Effects on Neural Networks | 604 |
| `demo_dropout_effect` | docs/modules/06-neural-networks.md | Numerical Example: Dropout Effect on Overfitting | 686 |
| `demo_early_stopping` | docs/modules/06-neural-networks.md | Numerical Example: Early Stopping in Action | 778 |

### module7_examples.py (6 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_fc_parameter_explosion` | docs/modules/07-computer-vision.md | Numerical Example: Parameter Explosion in Fully Connected Networks | 75 |
| `demo_convolution_by_hand` | docs/modules/07-computer-vision.md | Numerical Example: Convolution by Hand | 139 |
| `demo_output_size_formula` | docs/modules/07-computer-vision.md | Numerical Example: Output Size Formula in Action | 209 |
| `demo_cnn_vs_fc_parameters` | docs/modules/07-computer-vision.md | Numerical Example: CNN vs FC Parameter Comparison | 349 |
| `demo_pooling_dimension_tracking` | docs/modules/07-computer-vision.md | Numerical Example: Pooling Dimension Tracking | 278 |
| `demo_transfer_learning_comparison` | docs/modules/07-computer-vision.md | Numerical Example: Transfer Learning vs Random Features | 513 |

### module8_examples.py (8 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_tfidf_by_hand` | docs/modules/08-nlp.md | Numerical Example: TF-IDF Calculation by Hand | 130 |
| `demo_bow_sparsity` | docs/modules/08-nlp.md | Numerical Example: BoW Sparsity Problem | 73 |
| `demo_embedding_similarity` | docs/modules/08-nlp.md | Numerical Example: Embedding Similarity | 252 |
| `demo_vanishing_gradient` | docs/modules/08-nlp.md | Numerical Example: Vanishing Gradient | 403 |
| `demo_rnn_hidden_state` | docs/modules/08-nlp.md | Numerical Example: RNN Hidden State Evolution | 349 |
| `demo_self_attention` | docs/modules/08-nlp.md | Numerical Example: Self-Attention Step by Step | 537 |
| `demo_positional_encoding` | docs/modules/08-nlp.md | Numerical Example: Positional Encoding Patterns | 659 |
| `demo_bert_vs_gpt_scale` | docs/modules/08-nlp.md | Numerical Example: BERT vs GPT Scale Comparison | 854 |

### module9_examples.py (8 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_permutation_importance` | docs/modules/09-interpretability.md | Numerical Example: Permutation Importance Step by Step | 271 |
| `demo_partial_dependence` | docs/modules/09-interpretability.md | Numerical Example: Building a Partial Dependence Plot | 339 |
| `demo_shapley_game` | docs/modules/09-interpretability.md | Numerical Example: Shapley Values in a Simple Game | 392 |
| `demo_shap_sum_to_prediction` | docs/modules/09-interpretability.md | Numerical Example: SHAP Values Sum to Prediction | 473 |
| `demo_shap_summary_interpretation` | docs/modules/09-interpretability.md | Numerical Example: Reading a SHAP Summary Plot | 518 |
| `demo_lime_perturbation` | docs/modules/09-interpretability.md | Numerical Example: LIME Perturbation in Action | 577 |
| `demo_shap_vs_permutation` | docs/modules/09-interpretability.md | Numerical Example: SHAP vs Permutation Importance | 630 |
| `demo_shap_to_business` | docs/modules/09-interpretability.md | Numerical Example: From SHAP to Business English | 117 |

### module10_examples.py (5 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_fairness_metrics` | docs/modules/10-ethics-deployment.md | Numerical Example: Calculating Fairness Metrics by Hand | 217 |
| `demo_impossibility_theorem` | docs/modules/10-ethics-deployment.md | Numerical Example: Impossibility Theorem in Action | 145 |
| `demo_proxy_variable_correlation` | docs/modules/10-ethics-deployment.md | Numerical Example: Detecting Proxy Variables | 263 |
| `demo_roi_sensitivity_analysis` | docs/modules/10-ethics-deployment.md | Numerical Example: ROI Sensitivity Analysis | 649 |
| `demo_ab_test_sample_size` | docs/modules/10-ethics-deployment.md | Numerical Example: A/B Test Sample Size Calculation | 577 |

### deep_dive_data_prep_examples.py (2 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_feature_scaling_impact` | docs/appendices/data-preparation.md | Numerical Example: Feature Scaling Impact on k-NN | 96 |
| `demo_data_leakage` | docs/appendices/data-preparation.md | Numerical Example: Data Leakage Effect | 246 |

### deep_dive_evaluation_examples.py (2 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_confusion_matrix_metrics` | docs/appendices/model-evaluation.md | Numerical Example: Computing Metrics from Confusion Matrix Values | 86 |
| `demo_cv_variance_reduction` | docs/appendices/model-evaluation.md | Numerical Example: Why Cross-Validation Beats Single Splits | 186 |

### deep_dive_imbalanced_examples.py (2 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_precision_recall_threshold` | docs/appendices/imbalanced-data.md | Numerical Example: Precision-Recall at Different Thresholds | 46 |
| `demo_class_weights` | docs/appendices/imbalanced-data.md | Numerical Example: Effect of Class Weights | 119 |

### deep_dive_timeseries_examples.py (3 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_time_series_decomposition` | docs/appendices/time-series-forecasting.md | Numerical Example: Time Series Decomposition | 45 |
| `demo_arima_parameter_intuition` | docs/appendices/time-series-forecasting.md | Numerical Example: ARIMA Parameter Intuition | 112 |
| `demo_walk_forward_vs_kfold` | docs/appendices/time-series-forecasting.md | Numerical Example: Walk-Forward vs K-Fold Validation | 206 |

### deep_dive_cnn_examples.py (6 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_fc_vs_cnn_parameters` | docs/appendices/cnn-architecture.md | Numerical Example: FC vs CNN Parameter Comparison | 55 |
| `demo_weight_sharing_savings` | docs/appendices/cnn-architecture.md | Numerical Example: Weight Sharing Savings | 135 |
| `demo_receptive_field_growth` | docs/appendices/cnn-architecture.md | Numerical Example: Receptive Field Growth | 200 |
| `demo_convolution_verification` | docs/appendices/cnn-architecture.md | Numerical Example: Convolution Verification | 349 |
| `demo_output_size_formula` | docs/appendices/cnn-architecture.md | Numerical Example: Output Size Formula | 399 |
| `demo_pooling_dimensions` | docs/appendices/cnn-architecture.md | Numerical Example: Pooling Dimension Reduction | 540 |

### deep_dive_transformer_examples.py (8 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_positional_encoding` | docs/appendices/transformer-architecture.md | Numerical Example: Positional Encoding Values | 260 |
| `demo_embedding_lookup` | docs/appendices/transformer-architecture.md | Numerical Example: Embedding Lookup | 53 |
| `demo_scaling_effect` | docs/appendices/transformer-architecture.md | Numerical Example: Scaling Effect on Softmax | 427 |
| `demo_attention_scores` | docs/appendices/transformer-architecture.md | Numerical Example: Attention Scores Step by Step | 486 |
| `demo_multihead_reshape` | docs/appendices/transformer-architecture.md | Numerical Example: Multi-Head Reshape | 369 |
| `demo_ffn_forward` | docs/appendices/transformer-architecture.md | Numerical Example: FFN Forward Pass | 133 |
| `demo_layer_norm` | docs/appendices/transformer-architecture.md | Numerical Example: Layer Normalization | 185 |
| `demo_causal_masking` | docs/appendices/transformer-architecture.md | Numerical Example: Causal Masking | 767 |

### deep_dive_universal_approx_examples.py (7 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_sigmoid_weight_effects` | docs/appendices/universal-approximators.md | Numerical Example: Sigmoid Weight Effects | 46 |
| `demo_linear_regression_equiv` | docs/appendices/universal-approximators.md | Numerical Example: sklearn vs PyTorch Linear Regression | 118 |
| `demo_logistic_regression_equiv` | docs/appendices/universal-approximators.md | Numerical Example: sklearn vs PyTorch Logistic Regression | 194 |
| `demo_relu_bump_construction` | docs/appendices/universal-approximators.md | Numerical Example: ReLU Bump Construction | 330 |
| `demo_step_approximation` | docs/appendices/universal-approximators.md | Numerical Example: Step Function Approximation | 290 |
| `demo_polynomial_approximation` | docs/appendices/universal-approximators.md | Numerical Example: Learning x² with ReLU Networks | 392 |
| `demo_width_vs_accuracy` | docs/appendices/universal-approximators.md | Numerical Example: Width vs Accuracy Trade-off | 466 |

### deep_dive_surprising_phenomena_examples.py (7 functions)

| Function | Docs Page | Example | Line |
|----------|-----------|---------|------|
| `demo_bias_variance_tradeoff` | docs/appendices/surprising-phenomena.md | Numerical Example: Bias-Variance Tradeoff with Polynomial Regression | 42 |
| `demo_double_descent_random_features` | docs/appendices/surprising-phenomena.md | Numerical Example: Double Descent in Min-Norm Regression | 147 |
| `demo_interpolation_threshold_peak` | docs/appendices/surprising-phenomena.md | Numerical Example: The Interpolation Threshold Up Close | 207 |
| `demo_grokking_simulation` | docs/appendices/surprising-phenomena.md | Numerical Example: Grokking on Modular Addition | 272 |
| `demo_weight_norm_evolution` | docs/appendices/surprising-phenomena.md | Numerical Example: Weight Norm Evolution | 324 |
| `demo_emergence_metric_mirage` | docs/appendices/surprising-phenomena.md | Numerical Example: Metric Choice and Apparent Emergence | 389 |
| `demo_phase_transitions_comparison` | docs/appendices/surprising-phenomena.md | Numerical Example: Comparing Phase Transitions | 466 |

---

_Total demo functions mapped: 98._
