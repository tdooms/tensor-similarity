In-Context Linear Regression Implementation Spec
This document defines the task, model, training setup, checkpointing, and evaluation metrics for the in-context linear regression experiment.
The goal is to train a transformer to perform linear regression from examples provided in context. The model is evaluated across training checkpoints using behavioral metrics, OOD robustness metrics, attention-mass metrics, and residual/readout-geometry metrics.
---
1. Task
Each training sequence is one synthetic linear regression episode.
For each episode, sample a hidden task vector
[
t \sim \mathcal{N}(0, I_D).
]
For each context position (k = 1,\dots,K), sample an input vector
[
x_k \sim \mathcal{N}(0, I_D),
]
and generate a scalar label
[
y_k = t^\top x_k + \epsilon_k,
\qquad
\epsilon_k \sim \mathcal{N}(0,\sigma^2).
]
The model never observes (t). It only sees previous input-label examples and must predict the current label.
---
2. Data
Use the following default task/data parameters.
Parameter	Meaning	Value
(D)	Input/task dimension	4
(K)	Maximum number of in-context examples	8
(\sigma^2)	Label noise variance	0.05
(B)	Batch size	256
(T)	Training steps	500,000
(N_{\text{test}})	Evaluation episodes	2,048
Training data is generated online. Every training batch contains freshly sampled task vectors, inputs, and labels. There is no fixed finite dataset.
Total training episodes for the default run:
[
B \times T = 256 \times 500{,}000 = 128{,}000{,}000.
]
---
3. Tokenization / Context Format
Each input-label pair is represented using two tokens in (\mathbb{R}^{D+1}).
The input token stores (x_k) and sets the label coordinate to zero:
[
z^x_k =
\begin{pmatrix}
0 \
x_k
\end{pmatrix}
\in \mathbb{R}^{D+1}.
]
The label token stores (y_k) in the first coordinate and zeros elsewhere:
[
z^y_k =
\begin{pmatrix}
y_k \
0 \
\vdots \
0
\end{pmatrix}
\in \mathbb{R}^{D+1}.
]
A full sequence is
[
\mathrm{BOS}, z^x_1, z^y_1, z^x_2, z^y_2, \dots, z^x_K, z^y_K.
]
With (K=8), the sequence length is
[
1 + 2K = 17.
]
Predictions are made only at the (x_k) token positions. The (y_k) token is included so that it can be used as context for later predictions, but it is not used to predict itself because of causal masking.
For prediction (k), the available context is
[
\mathrm{BOS}, z^x_1, z^y_1, \dots, z^x_{k-1}, z^y_{k-1}, z^x_k.
]
The target is (y_k).
---
4. Embedding
The raw tokens are in (\mathbb{R}^{D+1}), where (D+1=5).
Use a learned input embedding/projection
[
W_E: \mathbb{R}^{D+1} \to \mathbb{R}^{d_{\text{model}}}.
]
Use a learned BOS token in (\mathbb{R}^{d_{\text{model}}}).
No learned positional embedding is used. Positional information is handled by RoPE inside attention.
---
5. Architecture
Use a decoder-only transformer with bilinear attention and bilinear MLP blocks.
Parameter	Meaning	Value
(L)	Number of transformer layers	2
(H)	Number of attention heads per layer	4
(d_{\text{model}})	Model width	64
(d_{\text{head}})	Head dimension	(d_{\text{model}}/H = 16)
(d_{\text{mlp}})	MLP hidden width	64
5.1 Positional encoding
Use RoPE in the attention layers.
5.2 Normalization
Do not use:
layer normalization,
RMSNorm,
standard per-token normalization.
Use BOS scalar normalization only pre-unembedding.  A reference is available at  workspaces/mel/bilinear_attn/experiments/norm_sweep/norms.py. Look for: Tok1 (lines 82-94).
Let (h_0) be the BOS hidden state immediately before the unembedding/readout stage. Define the scalar
[
s =
\sqrt{
\frac{1}{d_{\text{model}}}
\sum_{j=1}^{d_{\text{model}}}
h_{0,j}^2
+
\epsilon
}.
]
Before the final readout, divide all sequence hidden states by this scalar:
[
h_i \leftarrow \frac{h_i}{s}.
]
This gives one scalar normalization factor for the whole sequence.
5.3 Attention
Use bilinear attention.
For each head, compute two query/key projections:
[
Q_1, K_1, Q_2, K_2.
]
The pre-value attention interaction matrix is
[
A = (Q_1 K_1^\top) \odot (Q_2 K_2^\top),
]
where (\odot) is elementwise multiplication.
Apply a causal mask so that query position (q) can only interact with key positions (k \le q).
Use no Q/K attention biases.
5.4 MLP
Use a bilinear MLP block. The intended form is
[
\mathrm{BilinearMLP}(h)
W_O\left((W_L h) \odot (W_R h)\right),
]
with appropriate learned projections.
---
6. Readout and Loss
For each prediction position (x_k), read a scalar prediction from the corresponding hidden state:
[
\hat y_k = w_{\text{out}}^\top h_{x_k} + b_{\text{out}}.
]
The per-position loss is
[
\ell_k(w)
\mathbb{E}\left[(\hat y_k-y_k)^2\right].
]
The training loss averages over all prediction positions:
[
\ell(w)
\frac{1}{K}
\sum_{k=1}^{K}
\ell_k(w).
]
Use all (x_k) positions for training loss and for residual/readout structural metrics.
---
7. Training
Use online-generated synthetic batches.
Hyperparameter	Value
Optimizer	Muon
Peak learning rate	0.25
Weight decay	0.02
LR schedule	cosine decay
Warmup	1% of total training steps
Final LR	(0.2 \times) peak LR
Batch size	256
Training steps	500,000
Evaluation episodes	2,048
The learning rate warms up for the first 1% of training steps, then follows cosine decay until the final learning rate reaches (0.2) times the peak learning rate.
---
8. Checkpointing
Save 200 checkpoints per training run:
100 log-spaced checkpoints,
100 linearly spaced checkpoints.
Deduplicate checkpoint steps if log-spaced and linearly spaced schedules overlap.
Checkpoints are used for all behavioral, OOD, attention, embedding, and residual/readout metrics.
---
Evaluation Metrics
All metrics should be computed for every saved checkpoint unless otherwise stated.
---
9. In-Distribution Behavioral Metrics
9.1 Test loss
Mean squared error averaged over all validation episodes and all prediction positions:
[
\ell_{\text{test}}(w)
\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{E}\left[(\hat y_k-y_k)^2\right].
]
9.2 Per-position loss
For each (k=1,\dots,K), compute
[
\ell_k(w)
\mathbb{E}\left[(\hat y_k-y_k)^2\right].
]
Save all (K=8) losses. For plotting, selected positions such as (k=1,3,5,7) may be used.
9.3 ICL scores
Use two ICL scores:
[
ICL_{1:4} = \ell_4 - \ell_1,
]
[
ICL_{4:8} = \ell_8 - \ell_4.
]
Negative values mean the model improves with more context.
Since (D=4), (ICL_{1:4}) measures early-context improvement around the number of examples needed to identify a 4D linear task in the noiseless case. (ICL_{4:8}) measures later-context refinement.
9.4 Prediction magnitude / zero task-prior score
Since
[
t \sim \mathcal{N}(0,I_D),
]
the context-independent task-prior prediction is zero.
Track
[
\mathbb{E}\left[|\hat y_k|^2\right],
]
averaged over all prediction positions (k=1,\dots,K).
---
10. OOD Robustness Metrics
Use the following scale grid:
[
\log_{10}g \in {-1,-0.5,0,0.5,1,1.5,2,2.5}.
]
Equivalently,
[
g \in {0.1,0.316,1,3.16,10,31.6,100,316}.
]
Evaluate both OOD input scale and OOD task scale.
10.1 OOD input scale
For OOD input-scale evaluation, sample
[
x_k \sim \mathcal{N}(0,gI_D),
]
while keeping
[
t \sim \mathcal{N}(0,I_D).
]
For each (g), compute and store:
raw MSE,
normalized MSE,
prediction magnitude.
Raw MSE:
[
\mathrm{RawMSE}_x(g)
\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{E}\left[(\hat y_k-y_k)^2\right].
]
Normalized MSE:
[
\mathrm{NormMSE}_x(g)
\frac{\mathrm{RawMSE}_x(g)}{g^2}.
]
Prediction magnitude:
[
\mathbb{E}\left[|\hat y_k|\right],
]
averaged over validation episodes and prediction positions.
10.2 OOD task scale
For OOD task-scale evaluation, sample
[
t \sim \mathcal{N}(0,gI_D),
]
while keeping
[
x_k \sim \mathcal{N}(0,I_D).
]
For each (g), compute and store:
raw MSE,
normalized MSE,
prediction magnitude.
Raw MSE:
[
\mathrm{RawMSE}_t(g)
\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{E}\left[(\hat y_k-y_k)^2\right].
]
Normalized MSE:
[
\mathrm{NormMSE}_t(g)
\frac{\mathrm{RawMSE}_t(g)}{g^2}.
]
Prediction magnitude:
[
\mathbb{E}\left[|\hat y_k|\right],
]
averaged over validation episodes and prediction positions.
10.3 OOD ICL scores
For both OOD input-scale and OOD task-scale evaluations, also compute
[
ICL_{1:4}(g)=\ell_4(g)-\ell_1(g),
]
[
ICL_{4:8}(g)=\ell_8(g)-\ell_4(g).
]
Store these for each (g).
---
11. Embedding Structure Metrics
11.1 Embedding singular values
Compute the singular values of the input embedding matrix (W_E) at each checkpoint.
This tracks whether the embedding becomes effectively lower-dimensional during training.
---
12. Attention Mass Metrics
For each checkpoint, layer, head, validation episode, and query position, compute the bilinear attention interaction
[
A = (Q_1K_1^\top)\odot(Q_2K_2^\top).
]
Because (A) can be signed, convert it to causal attention mass:
[
p_{q,k}
\frac{|A_{q,k}|}
{\sum_{j \le q}|A_{q,j}|+\epsilon}.
]
All attention mass metrics are computed from (p).
Use query positions corresponding to all (x_k) prediction tokens.
Source positions can include:
BOS,
previous (x)-tokens,
previous (y)-tokens,
the current (x_k) token itself.
12.1 Attention mass entropy
For each query position (q), compute
[
H_q
-\sum_{k\le q}
p_{q,k}
\log p_{q,k}.
]
Lower entropy means the head concentrates interaction mass on fewer source tokens.
Optionally normalize by maximum entropy for the number of valid causal source positions:
[
\hat H_q =
\frac{H_q}{\log(q+1)}.
]
Use one convention consistently.
12.2 Attention mass variability
Measure how much the attention mass pattern changes across validation episodes.
For each layer/head/query position, let (\bar p_{q,k}) be the average attention mass over validation episodes. Then compute variability as the average distance from this mean pattern:
[
V_q =
\frac{1}{2}
\mathbb{E}{\text{episodes}}
\left[
\sum{k\le q}
\left|
p_{q,k} - \bar p_{q,k}
\right|
\right].
]
The factor (1/2) keeps the value in ([0,1]).
Low variability means the head uses a more fixed, input-independent pattern.
12.3 Previous-token attention mass
For each query position (q), track
[
p_{q,q-1}.
]
Average over validation episodes and query positions.
12.4 Previous-(x) attention mass
For each query position (q), sum mass assigned to previous input tokens:
[
\sum_{k \in \mathrm{previous}\ x\text{-tokens}} p_{q,k}.
]
Average over validation episodes and query positions.
12.5 Previous-(y) attention mass
For each query position (q), sum mass assigned to previous label tokens:
[
\sum_{k \in \mathrm{previous}\ y\text{-tokens}} p_{q,k}.
]
Average over validation episodes and query positions.
12.6 Total (x)-mass vs (y)-mass
Compute total attention mass assigned to all (x)-tokens and all (y)-tokens separately.
This tracks whether heads route information primarily from inputs or from labels.
---
13. Readout and Residual Geometry Metrics
All metrics in this section are computed using hidden states at all (x_k) prediction positions.
Let (h_{x_k}\in\mathbb{R}^{d_{\text{model}}}) be the hidden state at prediction position (x_k), after BOS scalar normalization and before the final scalar readout.
Collect these hidden states over all validation episodes and all (k=1,\dots,K). Center them:
[
h_c = h - \mathbb{E}[h].
]
Let (w_{\text{out}}) be the scalar readout vector and define the normalized readout direction
[
\hat w_{\text{out}}
\frac{w_{\text{out}}}{|w_{\text{out}}|}.
]
Do not center (w_{\text{out}}). It is a direction, not a sample cloud.
13.1 Readout-aligned variance fraction
Compute
[
\mathrm{RAV}
\frac{
\mathbb{E}\left[
(h_c^\top \hat w_{\text{out}})^2
\right]
}{
\mathbb{E}\left[
|h_c|^2
\right]
}.
]
This measures the fraction of residual-stream variation lying in the direction directly used to predict (y_k).
13.2 Readout-direction variance
Store the numerator as a debug metric:
[
\mathbb{E}\left[
(h_c^\top \hat w_{\text{out}})^2
\right].
]
13.3 Total residual variance
Store the denominator as a debug metric:
[
\mathbb{E}\left[
|h_c|^2
\right].
]
13.4 Final residual effective rank
Compute the covariance matrix of the centered hidden states:
[
\Sigma_h = \mathbb{E}[h_c h_c^\top].
]
Let (\lambda_i) be its eigenvalues. Normalize them:
[
p_i =
\frac{\lambda_i}
{\sum_j \lambda_j + \epsilon}.
]
Compute effective rank as
[
\mathrm{erank}
\exp\left(
-\sum_i p_i \log p_i
\right).
]
This measures whether the residual stream at prediction positions is spread across many directions or compressed into a lower-dimensional subspace.
---
14. Required Saved Outputs
For each checkpoint, save:
test loss,
all per-position losses (\ell_1,\dots,\ell_8),
(ICL_{1:4}),
(ICL_{4:8}),
prediction magnitude,
OOD input raw MSE for each (g),
OOD input normalized MSE for each (g),
OOD input prediction magnitude for each (g),
OOD input ICL scores for each (g),
OOD task raw MSE for each (g),
OOD task normalized MSE for each (g),
OOD task prediction magnitude for each (g),
OOD task ICL scores for each (g),
embedding singular values,
attention mass entropy by layer/head,
attention mass variability by layer/head,
previous-token attention mass by layer/head,
previous-(x) attention mass by layer/head,
previous-(y) attention mass by layer/head,
total (x)-mass and (y)-mass by layer/head,
readout-aligned variance fraction,
readout-direction variance,
total residual variance,
final residual effective rank.
---
15. Notes for Implementation
Use causal masking everywhere in the transformer and in attention-mass metric computation.
Predictions and losses are computed only at (x_k) token positions.
Do not compute loss at (y_k) token positions.
Use all (x_k) prediction positions for readout/residual metrics.
Use no Q/K attention biases.
Use RoPE, not learned positional embeddings.
BOS scalar normalization is applied only pre-unembedding.
Store raw OOD MSE even if normalized MSE is used for plotting.
Attention mass metrics use (|A|) because bilinear attention interactions can be signed.
---
16. Plotting Plan
The implementation should save all metrics listed above, but the paper figures should be organized around claims rather than plotting every scalar separately. The goal is to keep the main figures interpretable while preserving enough saved data for appendix/debug plots.
---
16.1 Figure group 1: In-distribution learning dynamics
This figure should show when the model learns the task, when in-context learning appears, and whether it first learns the zero task-prior prediction.
Recommended panels:
Panel A: Test loss and selected per-position losses
Plot on the same axes:
total test loss, labeled as mean test loss,
selected per-position losses, e.g. (\ell_1,\ell_3,\ell_5,\ell_7).
Save all (\ell_1,\dots,\ell_8), but only plot selected positions unless the full set is needed.
Panel B: ICL scores
Plot both ICL scores on the same axes:
[
ICL_{1:4} = \ell_4 - \ell_1,
]
[
ICL_{4:8} = \ell_8 - \ell_4.
]
Negative values indicate improvement from context.
Panel C: Prediction magnitude / zero task-prior score
Plot
[
\mathbb{E}\left[|\hat y_k|^2\right]
]
averaged over prediction positions.
This panel tests whether the model first learns the context-independent zero-prior prediction.
---
16.2 Figure group 2: OOD input-scale behavior
This figure should show how the model behaves when input magnitudes are shifted:
[
x_k \sim \mathcal{N}(0,gI_D).
]
Use the full (g)-grid:
[
\log_{10}g \in {-1,-0.5,0,0.5,1,1.5,2,2.5}.
]
Recommended panels:
Panel A: OOD input normalized MSE heatmap
Use training step on the x-axis and (g) on the y-axis.
Plot:
[
\mathrm{NormMSE}_x(g).
]
Panel B: OOD input ICL heatmap
Plot OOD ICL behavior over training and (g). Save both:
[
ICL_{1:4}(g),
\qquad
ICL_{4:8}(g).
]
If both are informative, use two panels. If one is clearly more useful, use the cleaner one in the main figure and move the other to appendix.
Panel C: OOD input prediction magnitude heatmap
Plot:
[
\mathbb{E}[|\hat y_k|]
]
over training and (g).
Raw MSE should be saved, but it does not need to be a main panel unless normalized MSE is misleading.
---
16.3 Figure group 3: OOD task-scale behavior
This figure should mirror the OOD input-scale figure, but with scaled task vectors:
[
t \sim \mathcal{N}(0,gI_D).
]
Recommended panels:
Panel A: OOD task normalized MSE heatmap
Plot:
[
\mathrm{NormMSE}_t(g).
]
Panel B: OOD task ICL heatmap
Save and inspect both:
[
ICL_{1:4}(g),
\qquad
ICL_{4:8}(g).
]
Use the clearer one in the main figure if space is limited.
Panel C: OOD task prediction magnitude heatmap
Plot:
[
\mathbb{E}[|\hat y_k|]
]
over training and (g).
This figure should allow direct comparison between input-scale robustness and task-scale robustness.
---
16.4 Figure group 4: Embedding structure
This figure should show whether the input embedding becomes lower-dimensional during training.
Recommended panel:
Panel A: Embedding singular values
Plot the singular values of the input embedding matrix over training.
Optional appendix/debug panel:
fraction of embedding variance explained by each singular direction.
---
16.5 Figure group 5: Attention hardening and routing
The model has (2) layers and (4) heads per layer, so there are only (8) heads. Save all attention mass metrics and inspect both summary and per-head views.
Recommended main figure:
Panel A: Attention mass entropy heatmap
Use heads on the y-axis and training step on the x-axis.
Lower values indicate harder / more concentrated attention mass.
Panel B: Attention mass variability heatmap
Use the same layout.
Lower values indicate more fixed, input-independent attention mass patterns.
Panel C: Previous-(x) attention mass heatmap
Shows whether heads route information from earlier input tokens.
Panel D: Previous-(y) attention mass heatmap
Shows whether heads route information from earlier label tokens.
Recommended supporting plots:
Line graph version  
Plot one line per head for entropy and/or variability. This is useful because there are only 8 heads.
Per-head attention pattern plots  
For heads that become interpretable, show their full attention-mass pattern. Use this to identify previous-token, previous-(x), previous-(y), or self-style heads.
Total (x)-mass vs (y)-mass  
Plot as a summary if it clarifies whether heads specialize toward input tokens or label tokens.
---
16.6 Figure group 6: Readout and residual compression
This figure should test whether the prediction-position residual stream becomes increasingly aligned with the scalar readout direction.
Recommended main panels:
Panel A: Readout-aligned variance fraction
Plot:
[
\mathrm{RAV}
\frac{
\mathbb{E}\left[
(h_c^\top \hat w_{\text{out}})^2
\right]
}{
\mathbb{E}\left[
|h_c|^2
\right]
}.
]
This is the main no-layernorm analogue of the layernorm/unembedding collapse analysis.
Panel B: Final residual effective rank
Plot effective rank of the centered prediction-position residual states.
This shows whether the residual stream becomes lower-dimensional.
Recommended debug panels:
Panel C: Readout-direction variance
Plot the numerator of RAV:
[
\mathbb{E}\left[
(h_c^\top \hat w_{\text{out}})^2
\right].
]
Panel D: Total residual variance
Plot the denominator of RAV:
[
\mathbb{E}\left[
|h_c|^2
\right].
]
The debug panels explain whether changes in RAV come from increased readout-direction variance, reduced total variance, or both.
Make sure not to use plt.show() or any annoying things that cannot run in tmux. 
