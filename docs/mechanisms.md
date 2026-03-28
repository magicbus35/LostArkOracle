# Lost Ark Economic Mechanisms (Knowledge Base)

This document defines the core logic for analyzing market impact. The LLM acts as an expert economist applying these rules to new events.

## 1. Succession Mechanism (Gear Transfer)
**Trigger**: When a new Raid allows "Gear Succession" (Transfer to higher tier/grade gear).

**Reaction Chain**:
1.  **Weapon First**: Players prioritize Attack Power. First clears lead to Weapon Succession.
    *   **Impact**: `Destruction Stones` (Weapon Material) demand spikes *first*.
    *   **Timing**: Depends on Clear Time (see below).
2.  **Armor Second**: After Weapon is done, players upgrade Armor for defense/HP.
    *   **Impact**: `Guardian Stones` (Armor Material) demand spikes *after* Weapon demand stabilizes/finishes.
    *   **Timing**: Often 1-2 weeks later than Destruction Stones.
3.  **Fusion Material**:
    *   **New Material?**: If update adds "Upper/New Fusion Material", *that* specific item skyrockets.
    *   **No New Material?**: The *highest existing* Fusion Material demand skyrockets.

## 2. Raid Difficulty & Clear Time Mechanism
The "Peak Timing" of material prices depends entirely on **how hard** the raid is.

**Case A: Extreme Difficulty (e.g., The First / Nightmare / Hell)**
*   **User Behavior**:
    *   Most users fail mid-week.
    *   "Try-hours" extend from Wed -> Sun -> Mon/Tue.
    *   Many give up and run Lower Difficulty (Hard -> Normal) at the massive deadline (Tuesday).
*   **Market Impact**:
    *   **Week 1**: Demand is *suppressed* or *slow* because few people clear enough to craft/succeed gear. They are busy wiping.
    *   **Week 2+**: As strategies solidify, clears become faster -> Succession happens -> **Huge Demand Spike**.
*   **Recommendation**:
    *   **Week 1**: Do not panic sell.
    *   **Week 2**: Sell when standard "Reclear" parties form.

**Case B: Moderate/Standard Difficulty (e.g., Normal Mode, Farm Status)**
*   **User Behavior**:
    *   "Friday Payday": Working adults clear raids on Friday night / Saturday.
    *   "Wednesday Resets": Hardcore users clear immediately on reset day.
*   **Market Impact**:
    *   **Wednesday**: Initial spike by aggregators/whales.
    *   **Friday/Saturday**: **Massive Volume Spike** (The Peak) as general population clears and hones.
    *   **Tuesday**: Prices dip as week ends.

## 3. General Principles
*   **In-Game Events are Short-Term**: Prices spike on expectation or immediate need, then normalize rapidly as supply catches up. "Short-selling" (Dan-ta) is the optimal strategy.
*   **Supply vs Demand**:
    *   Raid Release = Material CONSUMPTION (Demand Up).
    *   Raid Clear = Material PRODUCTION (Supply Up).
    *   We want to sell when Consumption > Production. (Usually Early Progression Phase).
