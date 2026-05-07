### SMART: Semantic Header Flattening and Pseudo-Code-Style Reasoning for LLM-based Complex Table Question Answering

Complex table question answering (TQA) remains challenging, as real-world table, usually
designed for human readability with multi-level
headers and fragmented hierarchical semantics, largely hindering large language models
(LLMs) from accurately aligning conditions,
attributes, and values during reasoning. Existing approaches typically rely on handcrafted
table linearization or prompts, forcing LLMs
to infer header hierarchies, which frequently
leads to brittle reasoning and hallucinations.
To this end, we propose SMART, a unified
framework that explicitly decouples table structure understanding from reasoning execution.
SMART consists of three components: Semantic Header Flattening for converting multi-level
headers into explicit single-level descriptors,
Global Understanding for capturing holistic
table-question semantics, and Pseudo-CodeStyle Reasoning for structured, step-by-step
inference with external validation. Extensive
experiments on multiple benchmarks demonstrate that SMART substantially improves both
the accuracy and robustness of complex TQA,
achieving state-of-the-art performance.