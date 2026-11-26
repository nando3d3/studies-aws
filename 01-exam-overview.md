# 01 — Exam Overview (AIF-C01)

## The certification

| Item | Value |
|---|---|
| Exam code | AIF-C01 |
| Name | AWS Certified AI Practitioner |
| Level | Foundational |
| Recommended prerequisite | AWS Cloud Practitioner (CLF-C02) — not required, but assumed knowledge |
| Delivery | Pearson VUE, in-person testing center or online proctored |
| Duration | 120 minutes of exam time (~150 minutes seat time with check-in, NDA, survey) |
| Questions | 65 total — **50 scored**, **15 unscored (pilot)** |
| Scoring | Scaled score 100–1000, **pass = 700** |
| Recertification | 3 years |
| Typical study time | ~10 hours average (5 if experienced, 20 if new) |

Unscored questions are indistinguishable from scored ones and carry no penalty. If a question feels impossibly obscure, it may well be a pilot question — answer and move on.

## Question formats

- Multiple choice (1 correct of 4)
- Multiple response (2+ correct of 5+)
- Ordering (arrange steps into sequence)
- Matching (pair items across two lists)
- Case study (one scenario, several questions)

The last three formats are newer to AWS exams. Practice them explicitly.

## Domains and weightings

| # | Domain | Weight |
|---|---|---|
| 1 | Fundamentals of AI and ML | 20% |
| 2 | Fundamentals of Generative AI | 24% |
| 3 | Applications of Foundation Models | 28% |
| 4 | Guidelines for Responsible AI | 14% |
| 5 | Security, Compliance, and Governance for AI Solutions | 14% |

Domains 2 and 3 together are just over half the exam, and they are overwhelmingly about **Amazon Bedrock**, prompt engineering, RAG, and fine-tuning. That is where your study time should concentrate.

## Explicitly out of scope

The exam guide states you will not be asked to:

- Develop or code ML algorithms
- Implement data engineering feature techniques
- Perform hyperparameter tuning implementation
- Build and deploy AI/ML pipelines
- Conduct mathematics or statistics

Expect conceptual and "which service fits this scenario" questions, not code.

## How to read scenario questions

Most service questions reduce to a matching exercise. Train yourself on these axes:

1. **What modality?** Text → Comprehend / Bedrock. Image or video → Rekognition. Documents → Textract. Speech out → Polly. Speech in → Transcribe. Language pairs → Translate.
2. **Managed AI service vs. build it yourself?** If the question mentions "no ML expertise," "fully managed," or "pre-trained," the answer is an AI service, not SageMaker.
3. **Custom model needed?** SageMaker for classical ML; Bedrock custom models / fine-tuning for foundation models.
4. **Grounding in private data?** RAG with Bedrock Knowledge Bases — not fine-tuning. Fine-tuning changes *behavior/style/task*; RAG supplies *facts*.
5. **Governance/responsible AI word appears?** Map to Clarify (bias/explainability), Model Monitor (drift), Guardrails (content filtering), A2I (human review), AI Service Cards (documentation), Model Cards (model documentation).

## Study plan (14 days, 1–2 h/day)

| Days | Focus |
|---|---|
| 1–2 | `02-ai-ml-fundamentals.md` — terminology, learning types, neural networks |
| 3–4 | `03-generative-ai-fundamentals.md` — transformers, tokens, embeddings, FM lifecycle |
| 5–7 | `04-amazon-bedrock.md` — the highest-value file. Read twice. |
| 8 | `05-sagemaker.md` |
| 9 | `06-aws-ai-services.md` |
| 10 | `07-data-analytics-and-stores.md` |
| 11 | `08-responsible-ai.md` + `09-security-compliance-governance.md` |
| 12 | `10-evaluation-metrics.md` + `11-cheatsheet.md` |
| 13–14 | `../questions/` — full question bank and the two practice exams |
