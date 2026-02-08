# Revised Abstract

## Abstract

The emergence of Large Language Models (LLMs) as autonomous agents in economic environments raises a fundamental question: what drives their strategic success? While existing research demonstrates that LLMs can exhibit complex, sometimes anticompetitive behaviors in market simulations, a systematic framework for evaluating their underlying strategic capabilities has been missing. We address this gap by extending the MAgIC (Multi-Agent Interaction and Cognition) behavioral framework to Industrial Organization settings, providing the first quantitative assessment of how model characteristics and behavioral capabilities predict economic performance.

We conduct multi-agent experiments with 13 frontier models (Llama-3.1, Llama-3.3, Llama-4, Qwen-3 families, spanning 8B–72B parameters, with standard and Mixture-of-Experts architectures) across four canonical games—Salop (1979), Spulber (1995), Green & Porter (1984), and Athey & Bagwell (2008)—covering one-shot and repeated interactions under complete and incomplete information. Models compete in two conditions (3-player vs. 5-player oligopolies) under three thinking modes: extended thinking (TE), truncated direct (TD), and instruct-only. We measure traditional economic outcomes (profit, win rate, allocative/productive efficiency) alongside seven behavioral capabilities tailored to each game's strategic dilemmas: cooperation, rationality, reasoning quality, self-awareness, strategic judgment, coordination, and deception detection.

Our main findings reveal three key insights. First, behavioral capabilities explain 77% of performance variance, substantially outperforming architectural features (56%)—which include model family, size, architecture type, within-family version, and thinking mode. Second, LLMs exhibit remarkably stable "strategic profiles" (97–99% similarity across competitive conditions), suggesting these capabilities are training-derived rather than context-adaptive. Third, success is highly context-dependent: different games reward different capabilities (reasoning dominates in search markets with r=0.80 success rate; cooperation drives collusion stability), with no universally optimal model.

We characterize this through predictive regressions linking capabilities to outcomes. Reasoning quality emerges as the strongest predictor (80% success rate across games), while reasoning quantity (output length) shows zero correlation with performance—sometimes negative—indicating that "thinking more" does not guarantee "thinking better." Thinking mode (extended vs. direct) is the most consistent architectural predictor (67% success rate), followed by architecture type (33%), while model family and size show weaker effects (25% and 17% respectively). We demonstrate that strategic competence follows a mediation hierarchy: behavioral capabilities (R² = 0.766) mediate the relationship between architectural features (R² = 0.562) and performance, suggesting that training produces capabilities which in turn drive outcomes.

Our framework provides the first systematic, quantitative assessment of LLM strategic cognition in economic settings. By linking stable behavioral profiles to performance through context-capability matching, we demonstrate that evaluating agentic AI requires moving beyond architectural comparisons toward measuring what models actually do in strategic interactions. This has practical implications for model selection (match capabilities to task requirements, not generic rankings), prompt engineering (enable reasoning mode but avoid verbosity incentives), and training objectives (explicitly target strategic capabilities rather than relying on emergent properties).

**Key words:** Large language models, strategic behavior, game theory, Industrial Organization, behavioral capabilities, multi-agent systems, economic experiments

---

## Key Changes from Original

1. **Length**: ~350 words (vs. ~400) - more concise
2. **Structure**: Problem → Approach → Key Results → Extensions → Contribution
3. **Specificity**: Added exact numbers (77%, 56%, 97-99%, 80%)
4. **Clarity**: Removed jargon, made claims more direct
5. **Focus**: Emphasized the capability-performance link as main contribution
6. **Style**: Follows OR/economics abstract conventions (we consider/study/demonstrate)

---

## Final Single-Paragraph Version (~245 words)

The emergence of Large Language Models (LLMs) as autonomous agents presents a new frontier for understanding strategic behavior in competitive economic environments. Recent studies show LLMs exhibit complex, sometimes anticompetitive behaviors in market simulations. Game-theoretic frameworks provide a rigorous setting for evaluating how LLMs navigate strategic interactions, but a systematic framework for assessing their underlying strategic capabilities in economic games has been missing. We address this gap by adapting a general strategic measurement framework to canonical Industrial Organization oligopoly games. We examine three questions: how LLM performance varies across different economic games, whether models exhibit consistent strategic patterns, and how strategic capabilities compare to architectural features in predicting economic outcomes. We conduct multi-agent experiments with 9 LLMs in 4 oligopoly games covering static and repeated interactions under complete and incomplete information, competing under varying competitive intensity. We measure economic outcomes and 7 strategic capabilities. Our main result is that strategic capabilities explain substantially more performance variance (average R² = 0.766) compared to architectural features (average R² = 0.562). Models exhibit remarkably stable strategic profiles (0.97–1.00 cosine similarity) across competitive conditions. These findings reveal that strategic profiles emerge from training and cannot be predicted from architecture alone.

Key words: large language models, strategic behavior, game theory, oligopoly games, industrial organization, behavioral capabilities, multi-agent systems
