# Project Log: Lost Ark Market Oracle (LMO)

## 2026-03-16: Crawler Automation & Explainable AI (XAI) UI Integration

### 1. Unified Auto-Trigger Pipeline
- **Action**: Integrated `subprocess` into the `crawl_notices.py` crawler logic to establish an autonomous trigger system.
- **Impact**: If even a single new notice is detected, the system now automatically chain-executes the downstream pipeline (Data Scraping -> LLM Economic Analysis -> Market Feature Extraction -> XGBoost Model Retraining). Achieved **Full Automation** where the entire process completes in 1-2 minutes without human intervention.

### 2. Streamlit Dashboard UI/UX & XAI Enhancements
- **Visualizing AI Reasoning**: Shifted from merely displaying numerical prediction charts to actively explaining the AI's logic. Newly added dedicated container boxes beneath the Sector Impact Correlation graphs explicitly output the LLM's `mechanisms_applied` reasoning and parameter weights.
- **Forced Korean LLM Output**: Modified the backend prompt in `analyze_events_llm.py`. The AI is now instructed via prompt engineering to output its economic analysis and logic strictly in the Korean language.
- **Translation Filtering Bug Fix**: Translated English tags in the dashboard ('Stream Summary', 'Roadmap Item') to user-friendly Korean ('방송 요약', '로드맵 예고'). Resolved the internal filtering conflict in the Inven tab caused by this translation by applying multi-condition (`isin`) regex masking.
- **Overall Impact**: Investors (Users) can now simultaneously read the charts alongside the AI's rationale (e.g., Honing Material Supply/Demand weights), drastically elevating the **Trust & Transparency** which is the core of quantitative modeling.

## 2026-01-16: Event Data Enrichment & Modeling

### Objective
Enrich official event data with market impact analysis to improve price prediction models.

### Actions Taken
1.  **Data Collection (Scraping)**
    - Created `scripts/scrape_notice_content.py`.
    - Scraped full HTML content from 450+ official Lost Ark notices (from `events_official.csv` links).
    - Identified CSS selector `div.fr-view` for main content.

2.  **LLM Enrichment (Perplexity API)**
    - Created `scripts/analyze_events_llm.py`.
    - Used Perplexity (`sonar-small-online`) to analyze notice text.
    - Extracted boolean flags: `Gold Nerf`, `New Raid`, `T4 Update`, `Gem Supply`, `New Package`.
    - Saved results to `data/events_enriched.csv`.

3.  **Feature Engineering**
    - Updated `src/process_data.py`.
    - Integrated extracted flags into the main training dataset (`lostark_features.csv`).
    - Created features: `is_gold_nerf`, `is_new_raid`, `is_t4`, `is_gem`, `is_package`.

4.  **Model Improvement**
    - Updated `scripts/train_model.py` to include new features.
    - Retrained XGBoost Regressor.
    - **Result**: RMSE improved from **58,987** to **57,869** (~1.9% improvement).

### 6. Script Reorganization (2026-01-16)
Restructured `scripts` folder for professional management:
- **`scripts/collection`**: Data ingestion (`getlostartdata.py`, `crawl_notices.py`).
- **`scripts/enrichment`**: Data enhancement (`analyze_events_llm.py`).
- **`scripts/modeling`**: Model logic (`train_model.py`).
- **`scripts/ops`**: Operations & Maintenance (`news_watcher.py`).
- **`scripts/utils`**: Tools & Ad-hoc scripts.

### Key Files
- `scripts/collection/scrape_notice_content.py`: Notice scraper.
- `scripts/enrichment/analyze_events_llm.py`: LLM analysis script.
### 7. Time-Travel Prevention & Walk-Forward Validation (2026-01 ~ 2026-02)
- **Rolling Window Cross-Validation**: Identified a critical data leakage error where the model was "peeking" at future data when backtesting past events. Implemented an "Honest Prediction" architecture enforcing strict walk-forward validation (retraining the model only on data prior to event N) to secure prediction reliability.

### 8. Two-Track Ingestion Architecture (Inven Community)
- **Community Sentiment Analysis**: Addressed the limitations of official patch notes (missing live stream context) by developing `crawl_inven.py`.
- **Filtering Logic**: Designed a Smart Year Inference (binary search) algorithm to rapidly pinpoint high-quality user summaries (e.g., from specific authors) immediately after broadcasts, filtering out noise.

### 9. Advanced Event Text Analysis (Hybrid Keyword Override)
- **Score Sensitivity Adjustment**: Discovered a critical flaw where LLM failed to understand the nuances of the Jan 10-14 updates, scoring them 0 despite a 70% spike in actual market demand.
- **Hybrid Logic**: Engineered a Python-based double safety net that detects specific trigger words (Reward, Recovery) and forces a High Score override, successfully preventing prediction flatlines.

### 10. Multi-Horizon Trajectory Prediction
- Upgraded the statistical model to an XGBoost `MultiOutputRegressor`, moving beyond simple +1 day predictions to output a full time-series trajectory from T-30 to T+30 based on the event date (T=0).
- Enhanced the dashboard visualization to help investors intuitively spot insights like "Sell at T+7".

### 11. Official API Integration & Bot Protection Bypass (2026-03)
Overcame the IP Ban limitations of scraping private data sites and finalized the project portfolio:
- **`scripts/collection/fetch_official_api.py`**: Integrated the official Lost Ark Developer API endpoint. Successfully bypassed Cloudflare 403 Forbidden bot protection to establish a legal and stable pipeline for fetching high-resolution 14-day market data.
- **Portfolio Draft Completed**: Documented core technologies including Walk-Forward Validation (preventing data leakage), LLM Hybrid Override, and Multi-Horizon time-series trajectory prediction (`portfolio_draft.md`).

---

## 2026-01-15: Initial Setup & Baseline Modeling

### Objective
Establish baseline data pipeline and model.

### Actions Taken
1.  **Market Data Collection**: Scraped `loachart.com` for item prices.
2.  **Event Collection**: Collected official notices list.
3.  **Baseline Model**: Trained initial XGBoost model using only basic event categories.

---

## Next Steps
- **Visualization**: Build Streamlit dashboard (`dashboard.py`) to monitor data and model.
- **Automation**: Schedule scraping and training.

---

## Project Evolution Footprints (Technical Milestones)

### 1. Event Data Intelligence (LLM Pipeline)
*   **v1.0 (Simple Keyword)**: Scraped official notices -> Extracted keywords (`Gold Nerf`, `New Raid`) -> Converted to Boolean features (`is_gold_nerf=1`). Reflected minimal market context.
*   **v2.0 (Quantitative Analysis)**: Introduced **Impact Score (1-10)** and **Sentiment (Positive/Negative)**.
    *   Goal: Teach the model not just *what* happened, but *how big* the impact was.
*   **v3.0 (Senior Economist Persona)**:
    *   **Chain of Thought**: Instructed LLM to "reason first, score later" to improve logic.
    *   **Noise Reduction**: Explicitly assigned **Score 0** to Routine Maintenance and simple text fixes (excluded from training).
    *   **Roleplay**: Prompted as "Loss Ark Senior Economist" to better interpret "Gold Value" vs "Item Price" correlations (e.g., Gold Nerf = Deflation = Material Prices Drop).

### 2. Data Engineering
*   **Unified Storage**: Merged scattered CSVs into `events_with_content.csv` (Notices) and `lostark_market_data_all.csv` (Price History).
*   **Optimization**: Applied `dtype` specifications to reduce memory usage significantly (mixed types handling).

---

## Critical System Constraints (DO NOT CHANGE)
1.  **Data Granularity**:
    *   **NEVER** compress the main market dataset (`lostark_features.csv`, `lostark_market_data_all.csv`) into Daily Averages.
    *   **Reason**: Short-term volatility (e.g., Friday Night Peaks, post-maintenance spikes) is the key to profit. 1-Day resolution hides these opportunities.
    *   *Allowed*: Creating temporary daily features for trend analysis is fine, but the source of truth must remain High-Frequency (5min/1hour).

## Future Roadmap (The "Oracle" Upgrade)
1.  **Community Sentiment Analysis (Inven Integration)**:
    *   **Trigger**: When an official notice announces "Live Stream (Ra-Bang)" or "LOA ON".
    *   **Action**: Scrape **Lost Ark Inven (10-Recommended Posts)** within 4 hours of the stream start time.
    *   **Selection Logic**: Smartly select posts with **High Recommendation Counts** AND **Detailed Content** (Long text length, keywords like 'Summary/요약'). Filter out short emotional reactions.
    *   **Goal**: Find high-quality community reactions and broadcast summaries.
    *   **Analysis**: Crawl the **FULL TEXT** of these posts (do not rely on 3-line user summaries) and analyze them with LLM to capture all detailed nuances, similar to official notice analysis.
    *   **Outcome**: Predict **BUY TIMING** (Accumulation Phase) to complement the current Sell Timing logic.
    *   *Logic*: "Buy early before the hype builds up, but not too early to kill ROI."

2.  **Official API Integration (Volume Data)**:
    *   **Goal**: Acquire 'Sales Volume' (Trade Count) to validate demand spikes.
    *   **Current Data**: Only has Price. Cannot distinguish "Price Up with Low Volume" (Manipulation) vs "Price Up with High Volume" (Real Demand).
    *   **Action**: Integrate Lost Ark Official API (`/markets/items`) to fetch daily/hourly trade counts.
    *   **Usage**: Use Volume as a confidence weight for the Prediction Model. (High Volume + High Price Forecast = Strong Buy).

## 2026-03-27: Integration of 'About Model' Portfolio Showcase into Dashboard
### Achievements
- **Dedicated Portfolio Showcase Tab**: Added the 'About Model' tab to the Streamlit Dashboard, transforming it from a simple monitoring tool into a comprehensive portfolio presentation for recruiters.
- **Visualizing Pipeline & Architecture (Plotly)**:
  - Implemented Funnel charts to illustrate the data reduction pipeline from 26.18M raw rows to 1,404 high-impact events.
  - Added Feature Importance charts showcasing the dominance of LLM-extracted qualitative metrics over standard time-series data.
  - Displayed Walk-Forward Validation architecture and RMSE optimization history using Bar and Line charts.
- **Domain Knowledge & Constraint Documentation**:
  - **Monotone Constraints**: Explicitly documented the use of monotone constraints in XGBoost to prevent logic errors (e.g., price dropping despite high demand). 
  - **Decision Translation**: Synthesized XGBoost logic into readable Flowcharts, Decision Rule Tables, and practical Prediction Scenarios.
- **UI Bug Fixes**: Resolved tuple unpacking ValueError caused by adding hist_preds to load_data() and eliminated duplicate Streamlit component rendering bugs.
