# 02 — Fundamentals of AI and ML (Domain 1, 20%)

## The nesting hierarchy

```
Artificial Intelligence
└── Machine Learning
    └── Deep Learning (neural networks)
        └── Generative AI
            └── Large Language Models
```

| Term | Definition |
|---|---|
| **Artificial Intelligence** | Machines performing tasks that mimic human behavior or normally require human intelligence. AI *simulates* intelligence (mimics behaviors); it does not *emulate* it (replicate the exact mechanism). |
| **Machine Learning** | Systems that improve at a task from data, without being explicitly reprogrammed for each new case. |
| **Deep Learning** | ML using artificial neural networks with multiple layers. Rule of thumb: **3 or more hidden layers = deep**. |
| **Generative AI** | AI that *creates* new content — text, image, audio, video, molecular structures — rather than only analyzing existing data. |
| **Large Language Model** | A foundation model built on the Transformer architecture that predicts the next token of text. |

**AI vs. Generative AI on the exam:** traditional AI *understands, classifies, and decides* over existing data; generative AI *produces new, previously unseen output*. Generative AI is a subset, not a replacement.

### Generative AI modalities

| Modality | Example output |
|---|---|
| Text | Articles, summaries, code |
| Vision | Images and video |
| Audio | Speech, music |
| Molecular | Candidate compounds for drug discovery |

**Multimodal** models handle more than one modality (e.g., image + text input).

## Learning problem types

### The three primary types

| Type | Data | Driven by | Typical tasks |
|---|---|---|---|
| **Supervised** | Labeled | Task | Classification, regression |
| **Unsupervised** | Unlabeled | Data | Clustering, association, dimensionality reduction |
| **Reinforcement** | No pre-existing dataset | Reward/feedback | Game playing, robotics, autonomous navigation |

Supervised learning is more accurate but requires expensive labeling up front. Unsupervised learning finds structure for free but needs a human to interpret whether the discovered patterns mean anything.

Reinforcement learning uses an **agent** acting in an **environment**, receiving **rewards** or penalties, learning by trial and error. There is no training dataset to begin with.

### Hybrid learning

| Type | Description |
|---|---|
| Semi-supervised | A small labeled set plus a large unlabeled set |
| Self-supervised | The model generates its own labels from the data (how LLMs are pre-trained) |
| Multi-instance | Examples are grouped into labeled "bags" rather than individually labeled |

### Learning techniques

| Technique | Description |
|---|---|
| **Transfer learning** | A model trained on one task is reused as the starting point for a related task. This is the conceptual basis of fine-tuning. |
| **Ensemble learning** | Several models are trained and their predictions combined (bagging, boosting, stacking). Useful when data quality is a problem. |
| **Active learning** | The model queries a human during training when it is uncertain. |
| **Online learning** | The model updates continuously as new data arrives. |
| **Multitask learning** | One model addresses several related problems at once. |

### Statistical inference

| Type | Direction |
|---|---|
| Inductive | Specific evidence → general rule |
| Deductive | General rule → specific outcome |
| Transductive | Specific inputs → specific predictions, without forming a general rule |

## The four classical ML problem shapes

Memorize this table — it recurs constantly.

| Problem | Labeled? | Output | Example | Algorithms |
|---|---|---|---|---|
| **Regression** | Yes | Continuous number | Predict tomorrow's temperature; house price | Linear, polynomial, SVR, decision tree, random forest regression |
| **Classification** | Yes | Discrete class | Spam vs. not spam; rain vs. sun | Logistic regression, KNN, SVM, Naive Bayes, decision tree, random forest, neural nets |
| **Clustering** | **No** | Groups | Segment customers into behavioral cohorts | K-means, DBSCAN, K-modes |
| **Association** | No | Relationships between variables | "People who bought bread also bought butter" | Apriori, FP-growth |

**Dimensionality reduction** is a fifth, usually a preprocessing step: shrink the number of features while preserving information. Algorithms: PCA, LDA, SVD, t-SNE.

Distinguishing trap: if the question says the data has *no labels*, the answer is clustering or association, never classification or regression.

## Divisions of ML by problem shape

| Division | Use when | Examples |
|---|---|---|
| Classical ML | Data is simple, features are clear, cost matters | Supervised, unsupervised |
| Reinforcement learning | No data exists; the system learns by acting | Game AI, robot navigation |
| Ensemble methods | Data quality is poor | Bagging, boosting, stacking |
| Neural networks / deep learning | Data is complex, features are not obvious | CNN, RNN, GAN, MLP, autoencoders |

## Neural networks

A neural network is a graph of **neurons (nodes)** arranged in layers, connected by **weighted** edges.

```
Input layer → Hidden layer(s) → Output layer
```

| Concept | Definition |
|---|---|
| **Feed-forward network (FNN)** | Connections never form a cycle; data flows one direction |
| **Backpropagation** | After a forward pass, error is propagated backwards to adjust weights. This is how the network learns. |
| **Loss function** | Compares prediction to ground truth to produce an error value |
| **Activation function** | Applied at a node to determine its output signal |
| **Dense layer** | Layer with increasing node count |
| **Sparse layer** | Layer with decreasing node count |
| **Perceptron** | The simplest neural network — a binary classifier with an input layer and an output layer |

The input layer does not transform data; it is only an entry point. The number of input nodes equals the dimensionality of the input vector.

### Activation functions

Linear activation functions cannot support backpropagation and cannot be stacked. Non-linear ones can — which is why every practical network uses them.

| Function | Range | Notes |
|---|---|---|
| Linear / Identity | Unbounded | No learning benefit; derivative is constant |
| Binary step | 0 or 1 | Earliest; binary classification only; rarely used now |
| **Sigmoid** | 0 to 1 | Smooth S-curve; suffers **vanishing gradient**; dense activation, slow |
| **Tanh** | −1 to 1 | Scaled sigmoid, stronger gradients, still vanishes |
| **ReLU** | 0 to ∞ | Fast, sparse activation, the default choice; suffers **dying ReLU** (neurons stuck at 0) |
| Leaky ReLU | −∞ to ∞ | Small negative slope prevents dying ReLU |
| Parametric ReLU | — | Leaky ReLU with a learned/fixed negative slope (commonly 0.01) |
| ReLU-6 | 0 to 6 | ReLU capped on the positive side |
| ELU | −1 to ∞ | Exponential on the negative side; faster convergence |
| Swish | — | Google Brain; smooth, non-monotonic ReLU alternative |
| Maxout | — | Returns the max of several inputs; doubles parameters |
| **Softmax** | 0 to 1, sums to 1 | Output layer for multi-class classification; produces a probability distribution |

Two named problems to recognize:
- **Vanishing gradient** — gradients shrink toward zero in deep networks, so early layers stop learning. Associated with sigmoid and tanh.
- **Dying ReLU** — neurons output zero permanently and stop contributing.

## Models, parameters, and inference

| Term | Definition |
|---|---|
| **Algorithm** | A set of instructions for performing a task |
| **Model** | A function that takes data, applies an algorithm, and produces a prediction |
| **Feature** | An individual measurable property extracted from raw data — a column in tabular data |
| **Feature engineering** | Extracting, cleaning, and transforming raw data into machine-readable features. ML models consume numbers, so text and categories must be encoded. |
| **Inference** | Requesting and receiving a prediction from a deployed model. Output typically includes a **confidence score**. |
| **Ground truth** | Correctly labeled data used as the objective standard for training and evaluation |

### Parameters vs. hyperparameters

This distinction is exam-guaranteed.

| | Parameters | Hyperparameters |
|---|---|---|
| Where | Internal to the model | External to the model |
| Set by | **Learned during training** | **Set manually before training** |
| Purpose | Make predictions | Control how training happens |
| Examples | Connection weights, biases | Learning rate, epochs, batch size, number of layers |

### Dataset splits

| Split | Purpose |
|---|---|
| **Training set** | The data the model learns from |
| **Validation set** | Used during training to tune hyperparameters and check generalization |
| **Test set** | Held out for a final, unbiased evaluation |

### Overfitting and underfitting

| Condition | Symptom | Typical fix |
|---|---|---|
| **Overfitting** | Excellent on training data, poor on new data — memorized noise | More data, regularization, simpler model, early stopping, dropout |
| **Underfitting** | Poor on both training and new data — model too simple | More complex model, more features, train longer |
| **Bias** | Systematic error from wrong assumptions | Usually paired with underfitting |
| **Variance** | Sensitivity to fluctuations in training data | Usually paired with overfitting |

### Model drift

A deployed model degrades over time as real-world data diverges from training data. This is **model drift** (leading to **model decay**). Amazon SageMaker Model Monitor detects it.

## Natural language processing

**NLP** sits at the intersection of computer science and linguistics: understanding the context of a **corpus** (a large structured body of text).

Pipeline vocabulary worth recognizing:

| Stage | Terms |
|---|---|
| Text wrangling | Case conversion, expanding contractions, stripping HTML and stop words, **tokenization**, **stemming**, **lemmatization** |
| Language understanding | Part-of-speech tagging, chunking, dependency parsing, constituency parsing |
| Processing tasks | **Named entity recognition (NER)**, n-grams, sentiment analysis, information extraction, question answering, topic modeling |

## Data concepts

### Data labeling and ground truth

Labeling is identifying raw data and attaching meaningful tags — a prerequisite for supervised learning. Amazon SageMaker Ground Truth is the managed service for this. Model accuracy can never exceed the quality of its ground truth.

### Data mining and CRISP-DM

**Data mining** extracts *patterns and knowledge* from large datasets — not the data itself.

**CRISP-DM**, the six-phase methodology:

1. Business understanding
2. Data understanding
3. Data preparation
4. Modeling
5. Evaluation
6. Deployment

### Knowledge mining

Using AI services to discover hidden insights, relationships, and patterns across large volumes of information. Three phases:

| Phase | What happens | AWS services |
|---|---|---|
| **Ingest** | Connect to structured (databases, CSV) and unstructured (PDF, video, audio) sources | S3, connectors |
| **Enrich** | Apply AI to extract information | Rekognition, Comprehend, Transcribe, Textract |
| **Explore** | Surface via search, BI, or feed to an LLM | Kendra, QuickSight, RAG |

Knowledge mining overlaps heavily with RAG.

### Data wrangling

Transforming raw data into a more valuable format. Six steps:

1. Discovery
2. Structuring
3. Cleaning (outliers, nulls, duplicates, formatting)
4. Enriching (join in additional context)
5. Validating (quality and reliability checks)
6. Publishing (write to a data store)

AWS tools: **SageMaker Data Wrangler** (ML-focused) and **AWS Glue DataBrew** (general-purpose).

### Data types

```
Qualitative (quality)
└── Categorical
    ├── Binary   — two values (true/false)
    ├── Nominal  — categories with no order (colors)
    └── Ordinal  — categories with order (small/medium/large)

Quantitative (quantity)
└── Numerical
    ├── Interval — no true zero (temperature in °C)
    └── Ratio    — true zero exists (weight, price)
```

### Data roles

| Role | Focus |
|---|---|
| Data analyst | Uses existing information to produce actionable business insight |
| Data scientist | Math, statistics, predictive modeling, ML — predicts the future |
| Data engineer | Builds the infrastructure and pipelines that move and store data |

## The AWS ML workflow

AWS documents the ML lifecycle as phases containing steps:

1. **Fetch** — collect data from a repository, public dataset, or source system
2. **Clean** — normalize inconsistent values
3. **Prepare** — transform and engineer features
4. **Train** — select an algorithm and train the model
5. **Evaluate** — measure accuracy against held-out data
6. **Deploy** — host the model for inference
7. **Continuously evaluate** — monitor for drift, collect ground truth, retrain

Expect ordering-format questions on this sequence.

## Jupyter ecosystem

| Tool | Description |
|---|---|
| Jupyter Notebook | Web documents combining live code, narrative text, equations, and visualizations |
| JupyterLab | The next-generation interface: notebooks plus terminal, editor, and file browser |
| JupyterHub | Multi-user server for teams and classes |
| SageMaker Studio | AWS's hosted JupyterLab environment |
| SageMaker Studio Lab | Free, standalone notebooks — **outside your VPC**, so unsuitable for regulated data |
