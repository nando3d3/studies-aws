# 03 — Fundamentals of Generative AI (Domain 2, 24%)

## Foundation models

A **foundation model (FM)** is a large, general-purpose model pre-trained on a vast and diverse dataset, designed to be adapted to many downstream tasks. Synonyms you will see used interchangeably: *base model*, *pre-trained model*.

A **large language model (LLM)** is a foundation model that implements the Transformer architecture and works primarily in the text modality. Not every foundation model is an LLM — Stable Diffusion is a foundation model for images.

### The foundation model lifecycle

1. **Data selection** — assemble a large, diverse, licensed corpus
2. **Pre-training** — self-supervised training on the corpus, producing a base model
3. **Fine-tuning / adaptation** — specialize for a task, domain, or behavior
4. **Alignment** — RLHF or similar, to make outputs helpful and harmless
5. **Evaluation** — benchmark accuracy, toxicity, robustness
6. **Deployment** — serve behind an endpoint
7. **Monitoring / feedback** — collect real usage, iterate

### Costs and trade-offs of foundation models

| Advantage | Trade-off |
|---|---|
| No training data or ML expertise required to get started | Inference cost scales with tokens |
| Broad capability across tasks | May hallucinate; no knowledge of private or recent data |
| Fast time to value | Limited control over the base model's behavior |
| Multimodal options available | Larger models cost more and respond slower |

Selecting a model is a trade-off among **cost, latency, modality, context window size, licensing, and task performance**. A small model like Claude Haiku is cheaper and faster; a large one like Opus is more capable. The exam favors "choose the smallest model that meets the requirement."

## Transformer architecture

Introduced by Google researchers in the paper *Attention Is All You Need*. It replaced RNNs and CNNs for NLP because it processes sequences in parallel rather than step by step, which is what made scaling to billions of parameters practical.

Two components:

| Component | Role |
|---|---|
| **Encoder** | Reads and builds a contextual understanding of the input |
| **Decoder** | Generates output tokens one at a time, conditioned on the encoder output and what it has already produced |

Two mechanisms make it work:

- **Multi-head attention** — many attention computations run in parallel, each capturing a different kind of relationship between tokens
- **Positional encoding** — because tokens are processed in parallel, position vectors are added to embeddings so the model knows word order

### Attention types

| Type | Description | Where |
|---|---|---|
| **Self-attention** | Every element in a sequence attends to every other element in the same sequence | Encoder |
| **Cross-attention** | Elements of one sequence attend to a different sequence | Decoder attending to encoder output (e.g. translation) |
| **Multi-head attention** | Several attention heads in parallel | Both |

Attention is computed with **Query, Key, Value (Q/K/V)** — conceptually a search: the query is matched against keys, and the matching values are returned weighted by relevance.

## Tokens

A **token** is a chunk of text — roughly a word or word fragment — mapped to a unique integer ID from the model's vocabulary. Typical vocabularies are 30,000 to 100,000+ tokens.

| Tokenizer | Used by |
|---|---|
| Byte Pair Encoding (BPE) | GPT-3 |
| WordPiece | BERT |
| SentencePiece | T5, GPT-3.5 |

Managed services like Bedrock tokenize for you.

**Why tokens matter commercially:** memory and compute both grow with sequence length, and providers bill per input token and per output token, usually at different rates. Input and output share one budget — a very long prompt leaves less room for the answer. The standard mitigation for long conversations is to summarize earlier turns and feed the summary forward.

### Context window

The **context window** is the total number of tokens a model can consider in one request — input plus output combined. Everything you inject (system prompt, retrieved documents, conversation history, few-shot examples) consumes it.

## Embeddings and vector space

An **embedding** is a numeric vector representing a piece of data (text, image, audio) such that semantically similar items land near each other in a high-dimensional space.

- A **vector** has magnitude and direction; a **vector space model** represents documents as vectors so that distance encodes similarity.
- **Dimensionality** is the number of values in the vector. The embedding model determines it, and it must match your vector store's index configuration. Example: `all-MiniLM-L6-v2` produces 384 dimensions.
- Different embedding models capture different relationships — spelling similarity, semantics, domain-specific meaning.
- Similarity is measured with **cosine similarity**, Euclidean distance, or dot product. Cosine is the common default.

Embedding models on Bedrock: **Amazon Titan Text Embeddings (v1, v2)**, **Cohere Embed English**, **Cohere Embed Multilingual**.

## Inference parameters

| Parameter | Effect |
|---|---|
| **Temperature** | Controls randomness. 0 is deterministic and repetitive; ~0.5 is a balanced default; 1.0 is maximally creative and can degrade into nonsense. |
| **Top-p (nucleus sampling)** | Considers only the smallest set of tokens whose cumulative probability exceeds p. Lower = more focused. |
| **Top-k** | Considers only the k most likely next tokens. |
| **Max tokens / response length** | Caps the length of the generated output. |
| **Stop sequences** | Strings that, when generated, end the response. |

For factual, deterministic tasks (classification, extraction, SQL generation) use a **low temperature**. For creative writing, raise it.

## Prompt engineering

### The three core techniques

These are the ones the exam cares about most.

| Technique | Definition | When to use |
|---|---|---|
| **Zero-shot** | Give the task with instructions but **no examples** | Simple, well-understood tasks |
| **Few-shot** | Provide **several input/output examples** before the real input | You need a specific output format or the task is ambiguous |
| **Chain of Thought (CoT)** | Ask the model to reason **step by step** before answering | Multi-step reasoning, math, logic. Helps smaller models disproportionately. |

*One-shot* is few-shot with exactly one example.

### Elements of a good prompt

- **Instructions** — what to do
- **Context** — background the model needs
- **Input data** — the thing to act on
- **Output indicator** — the required format
- **Negative prompting** — explicitly state what not to do ("do not add information not present in the text")

### Model-specific practices

Prompting is not universal across providers.

- **Anthropic Claude** responds strongly to **XML tags** delimiting sections (`<context>`, `<question>`, `<document>`). Claude also benefits notably from Chain of Thought.
- OpenAI-style models do not depend on XML tags in the same way.
- Different providers use different request formats entirely (covered in `04-amazon-bedrock.md`).

### Prompt risks

| Risk | Description |
|---|---|
| **Prompt injection** | User input contains instructions that hijack the model's behavior |
| **Jailbreaking** | Direct injection that overrides or reveals the system prompt |
| **Prompt leaking** | Coaxing the model into revealing its system prompt or confidential context |
| **Poisoning / hijacking** | Malicious content planted in retrieved documents or training data |

Mitigations: separate system instructions from user content, apply guardrails, restrict backend privileges, add human review, and validate anything retrieved from external sources.

## Retrieval Augmented Generation (RAG)

**RAG** retrieves relevant external data at query time and injects it into the prompt so the foundation model can answer from facts it was never trained on.

```
User question
  → Embed the question
  → Search the knowledge source for the most relevant chunks
  → Insert those chunks into the prompt as context
  → Foundation model generates a grounded answer
```

Key exam points:

- RAG **does not modify the model**. No weights change. It changes only what is in the context window.
- RAG is **not limited to vector stores**. A RAG pipeline can retrieve from the internet, a relational database, a graph database, or a document database. Vector search is simply the most common implementation for semantic retrieval.
- RAG is how you handle **private, proprietary, or frequently changing data**, and how you reduce hallucination by grounding answers in cited sources.
- The limit of RAG is the context window and the cost of the tokens you inject.

### RAG vs. fine-tuning vs. prompt engineering

The single most likely scenario question on this exam.

| Approach | Cost | Changes weights? | Use when |
|---|---|---|---|
| **Prompt engineering** | Lowest | No | The base model can already do it with better instructions |
| **RAG** | Low–medium | No | You need current, private, or citable **facts** |
| **Fine-tuning** | High | Yes | You need a specific **behavior, format, tone, or task** the base model does poorly |
| **Continued pre-training** | Highest | Yes | You need to teach a whole new **domain of general knowledge** |
| **Train from scratch** | Extreme | N/A | Almost never the right answer on this exam |

Mnemonic: **RAG adds knowledge, fine-tuning adds skill.**

## Fine-tuning

Fine-tuning retrains a pre-trained model's weights on a smaller, labeled dataset. The output is a new set of **model weights**, not new model code.

### By what the dataset teaches

| Type | Data | Purpose |
|---|---|---|
| **Instruction fine-tuning** | Prompt/completion pairs | Teach the model to follow a specific instruction pattern |
| **Domain-specific fine-tuning** | Domain corpus | Adapt vocabulary and knowledge to a vertical |
| **Continued pre-training** | **Unlabeled** data | Broaden general knowledge in a domain |
| **Supervised fine-tuning (SFT)** | **Labeled** data | Improve performance on a specific task |

**Labeled vs. unlabeled is the discriminator the exam uses:** fine-tuning = labeled data, task-specific. Continued (continuous) pre-training = unlabeled data, general knowledge.

### By how much of the model changes

| Method | Description |
|---|---|
| **Full fine-tuning** | Update every weight. Most expensive, most capable. |
| **Parameter-Efficient Fine-Tuning (PEFT)** | Freeze most of the model and train a small set of added parameters. Much cheaper. **LoRA** is the best-known PEFT method. |
| **Last-layer fine-tuning** | Freeze everything except the final layer |
| **Pruning** | Remove parameters to shrink the model |

### RLHF

**Reinforcement Learning from Human Feedback** aligns a model with human preferences: humans rank model outputs, a reward model is trained on those rankings, and the model is optimized against the reward model. This is what makes a raw pre-trained model into a helpful assistant. Related: Amazon A2I and Bedrock's human evaluation workflows provide the human judgment component.

## Agents

An **agent** is an LLM given autonomy to decide which actions to take. It can call **tools** (functions, APIs, Lambda), query knowledge bases, and chain multiple steps to satisfy a request.

| Concept | Meaning |
|---|---|
| **Tool use / function calling** | The model chooses to invoke an external function and supplies the arguments |
| **Agency** | The model decides the sequence of steps; you did not script it |
| **Orchestration / workflow** | You explicitly define the sequence; the model fills in the generative steps |

Agentic (Bedrock Agents) versus orchestrated (Bedrock Prompt Flows, LangChain chains) is a recurring comparison.

## Model architectures worth naming

| Architecture | Purpose |
|---|---|
| **Transformer** | Modern LLMs; attention-based |
| **GAN** (Generative Adversarial Network) | A generator and a discriminator compete; used for synthetic images |
| **VAE** (Variational Autoencoder) | Encodes data into a latent space and decodes it back; used in generation |
| **Diffusion model** | Iteratively denoises random noise into an image; how Stable Diffusion works |
| **CNN** | Convolutional networks for images |
| **RNN / LSTM** | Sequential networks, predecessors to the Transformer; limited by sequence length |

## Business use cases and metrics

Common generative AI applications: chatbots and virtual assistants, summarization, translation, code generation, content creation, image generation, search and knowledge retrieval, personalization, document processing, and customer service deflection.

Business metrics you may be asked to associate with a use case:

| Metric | Meaning |
|---|---|
| User satisfaction / CSAT | Are users happy with generated responses |
| Average revenue per user (ARPU) | Revenue impact of the feature |
| Cross-domain performance | Does the model hold up outside its primary domain |
| Conversion rate | Did the AI feature drive the desired action |
| Efficiency / cost per task | Reduction in human effort or cost |
| Time to market | Speed of shipping the capability |
| ROI | Overall return relative to spend |
