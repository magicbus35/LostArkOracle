# Neural-Symbolic Architecture: "The Brain & The Calculator"

This document defines the roles of the AI components in LMO (Lost Ark Market Oracle).

## 1. The Problem
- **LLMs (GPT/Sonnet)** are great at **reading** (Parsing patch notes, understanding "Nerf" vs "Buff") but terrible at **math** and **probability** (Predicting exact price movements).
- **Statistical Models (XGBoost/Prophet)** are great at **math** (Time-series trends) but cannot read text.

## 2. The Solution: Hybrid Pipeline

We use a **Neural-Symbolic** approach where the LLM converts unstructured text into structured "Features", which the ML Model then uses to calculate predictions.

### Step 1: Feature Extraction (The LLM's Job)
The LLM reads the Patch Notes and Stream Summaries. It does **NOT** predict the price. It only extracts **FACTS**.

| Input (Text) | LLM Output (Features) | Data Type |
| :--- | :--- | :--- |
| "Profound changes to T4..." | `is_t4_update` | `True` |
| "Director mentioned this in stream..." | `is_pre_announced` | `True` |
| "Nerfing gold rewards..." | `event_type` | `Gold Nerf` |
| "Sentiment seems negative..." | `sentiment_score` | `-0.8` |

### Step 2: Quantitative Prediction (The ML Model's Job)
The ML Model (XGBoost) takes these **Features** + **Market Data** to output a specific prediction.

**Input Vector**:
`[is_pre_announced=1, is_t4_update=1, current_volatility=high, market_trend=bullish]`

**Model Logic (Learned from History)**:
> *"I have seen 50 cases where `is_pre_announced=1` and `trend=bullish`. In 45 of them, the price DROPPED on update day (Sell the news)."*

**Output**:
`Predicted Price Change (Day 0): -3.2%`
`Confidence: 85%`

## 3. Workflow
1.  **Collection**: Scraper gets Inven Post + Official Note.
2.  **Enrichment**: `analyze_events_llm.py` (LLM) creates `events_enriched.csv` with `is_pre_announced` flag.
3.  **Modeling**: `train_model.py` reads `events_enriched.csv` + `market_prices.csv`.
4.  **Inference**: Dashboard loads the trained model. When user clicks "Analyze", it runs the pipeline to show: **"Expected Impact: -3.2% (Sell Signal)"**.
