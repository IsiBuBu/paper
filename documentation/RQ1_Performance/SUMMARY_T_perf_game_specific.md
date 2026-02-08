# Summary: Game-Specific Strategic Metrics

**Table:** `T_perf_game_specific.csv` | **Research Question:** RQ1 - Competitive Performance  
**Models:** 13 (12 LLMs + Random) | **Metrics:** 4 game-specific efficiency/behavior measures

---

## Data Overview

Beyond win rates and profits, this table captures **game-specific strategic behaviors** unique to each oligopoly structure. Metrics measure efficiency, coordination, and pricing strategies under 3P vs 5P conditions.

**Key Innovation:** Moves beyond aggregate performance to reveal **mechanism-specific competencies**

---

## Metric 1: Salop Market Price (↑ higher better)

**Definition:** Average equilibrium price in differentiated product market  
**Economic Meaning:** Higher prices indicate successful differentiation and market power maintenance

### Results
- **3P**: Mean=12.97, Range=[11.32–13.89]
- **5P**: Mean=12.98, Range=[11.53–13.79]
- **3P→5P Effect**: +0.01 (+0.1%), **p=0.2528 ns** — No significant change

### Top Performers (3P)
1. L3.3-70B: 13.89
2. L3.1-70B: 13.80
3. Q3-32B (TD): 13.66

### Bottom Performers (3P)
- L3.1-8B: 11.32 (significantly lower pricing power)
- Random: 11.61

**Interpretation:** Price levels **remarkably stable** across market structures. Models maintain differentiation strategies regardless of competition intensity. **No structural sensitivity.**

---

## Metric 2: Spulber Allocative Efficiency (↑ higher better)

**Definition:** How well market allocates goods to high-valuation buyers (0–1 scale)  
**Economic Meaning:** 1.0 = perfect matching, 0 = random allocation

### Results
- **3P**: Mean=0.516, Range=[0.35–0.74]
- **5P**: Mean=0.548, Range=[0.22–0.75]
- **3P→5P Effect**: +0.032 (+6.2%), **p=0.0079*** — **Significant improvement**

### Top Performers (5P)
1. L4-Scout: **0.75** (excellent allocation)
2. Q3-14B (TD): 0.74
3. L4-Maverick: 0.73

### Bottom Performers (5P)
- L3.3-70B: 0.22 (allocation failure)
- L3.1-8B: 0.22
- Q3-14B (TE): 0.28

**Interpretation:** **Counter-intuitive finding** — More players **improve** allocative efficiency despite lower profits. Suggests mechanism complexity benefits from more market data/transactions. **L4 models excel** at matching optimization.

---

## Metric 3: Green-Porter Reversion Frequency (↓ lower better)

**Definition:** Proportion of rounds in punishment/price-war phases  
**Economic Meaning:** Lower values indicate better collusion sustainability

### Results
- **3P**: Mean=0.131, Range=[0.127–0.137]
- **5P**: Mean=0.128, Range=[0.125–0.135]
- **3P→5P Effect**: -0.003 (-2.3%), **p=0.0036*** — **Significant reduction**

### Best Performers (3P, lowest reversion)
1. Q3-14B (TD): 0.127
2. Q3-235B Inst: 0.127
3. Multiple models: ~0.127

### Worst Performers (3P, highest reversion)
- Q3-32B (TE): 0.137
- L3.1-70B: 0.136
- L3.1-8B: 0.135

**Interpretation:** **Surprising result** — More players lead to **fewer** punishment phases. Classical theory predicts harder collusion with more players. Suggests LLMs may use less aggressive monitoring strategies or better tacit coordination in larger groups. **Low variance** (0.127–0.137) indicates universal collusion competence.

---

## Metric 4: Athey-Bagwell Productive Efficiency (↑ higher better)

**Definition:** Production efficiency given capacity constraints (0–1 scale)  
**Economic Meaning:** How well firms utilize their production capacity

### Results
- **3P**: Mean=0.538, Range=[0.289–0.595]
- **5P**: Mean=0.396, Range=[0.129–0.497]
- **3P→5P Effect**: -0.142 (-26.4%), **p<0.001*** — **Strong decline**

### Top Performers (3P)
1. L4-Scout: 0.595
2. L4-Maverick: 0.595
3. Q3-235B Inst: 0.594

### Bottom Performers (3P)
- Random: 0.289
- L3.1-8B: 0.347
- Q3-14B (TD): 0.397

**Interpretation:** **Largest efficiency drop** of all metrics. More players fragment market share, reducing capacity utilization. **L4 models dominate** production optimization. Decline explains Athey-Bagwell's significant 3P→5P profit drop despite stable win rates.

---

## Cross-Metric Structural Sensitivity Summary

| Metric | Direction | 3P→5P Change | P-Value | Structural Effect |
|--------|-----------|--------------|---------|-------------------|
| **Salop Price** | ↑ | +0.1% | 0.2528 ns | ❌ None |
| **Spulber Efficiency** | ↑ | **+6.2%** | 0.0079*** | ✅ **Improves** |
| **Green-Porter Reversion** | ↓ | **-2.3%** | 0.0036*** | ✅ **Improves** |
| **Athey-Bagwell Efficiency** | ↑ | **-26.4%** | <0.001*** | ✅ **Worsens** |

**Pattern:** **3/4 metrics show structural sensitivity**, but effects diverge:
- **Athey-Bagwell**: Capacity fragmentation hurts efficiency
- **Spulber**: More transactions improve allocation matching
- **Green-Porter**: Better tacit coordination with more players
- **Salop**: Differentiation strategy robust to market size

---

## Model-Specific Strategic Profiles

### L4 Models (Scout, Maverick)
- **Excel at**: Spulber allocation (0.73–0.75), Athey-Bagwell production (0.595)
- **Specialization**: Optimization and matching tasks
- **Weakness**: Inconsistent win rates (0–100% range)

### Q3-14B (TE)
- **Excel at**: Overall win rate (87.5%), Green-Porter stability (0.127 reversion)
- **Weakness**: Spulber allocation drops in 5P (0.28, worst among top models)
- **Profile**: Strategic dominance but allocation mechanism struggles

### Q3-32B (TE)
- **Excel at**: Salop pricing (13.66), profit extraction (2,215 avg)
- **Weakness**: Green-Porter reversion (0.137, highest)
- **Profile**: Aggressive pricing, less collusion-friendly

### L3.3-70B & L3.1-70B
- **Excel at**: Salop pricing (13.89, 13.80)
- **Weakness**: Spulber allocation (0.22), complete Salop/Spulber win failures
- **Profile**: Size advantage in simple pricing, fails in complex mechanisms

---

## Key Takeaways

1. **Structural effects are metric-specific** — 3/4 show significant 3P→5P changes, but directions vary
2. **Counter-intuitive improvements exist** — Spulber efficiency and Green-Porter collusion **improve** with more players
3. **Athey-Bagwell efficiency drop explains profit decline** — -26% efficiency → -5% profit despite stable win rates
4. **Model specializations emerge** — L4 excels at optimization, Q3 at strategy, Llama at simple pricing
5. **Low strategic variance in collusion** — All models competent at Green-Porter (0.127–0.137 range)
  - Q3-14B TE: 0.595
  - Q3-235B Inst: 0.595
  - Multiple TE models: 0.592–0.595
- **Worst (3P):**
  - Q3-14B TD: 0.289 (dramatically lower)
- **Top performers (5P):**
  - All TE models: ~0.497
  - Q3-32B TD: 0.457

**Interpretation:** More players → LESS efficient production. This confirms capacity utilization becomes harder with more competitors. Clear **TE > TD advantage** in productive efficiency.

## Key Statistical Findings

### Significance Summary

| Game | Metric | Direction | P-Value | Effect Size |
|------|--------|-----------|---------|-------------|
| Salop | Market Price | None | 0.2528 | No change |
| Spulber | Allocative Efficiency | ↑ Increase | 0.0079 ** | +0.02–0.08 |
| Green-Porter | Reversion Frequency | ↓ Decrease | 0.0036 ** | -0.002 |
| Athey-Bagwell | Productive Efficiency | ↓ Decrease | <0.001 *** | -0.095 |

### Counter-Intuitive Findings

1. **Spulber Allocative Efficiency INCREASES with more players**
   - Contradicts profit data (profits decline in 5P)
   - Suggests: Better market outcomes, but individual payoffs suffer
   - Models like L4-Scout excel at matching despite low profits

2. **Green-Porter Reversion Frequency DECREASES with more players**
   - Expected: More players → harder to coordinate → more punishment
   - Observed: More players → LESS punishment
   - Possible explanation: Models become more cautious or cooperative

3. **Salop Prices STABLE despite competition increase**
   - Expected: More competition → lower prices
   - Observed: No significant price change
   - Suggests: Product differentiation maintains pricing power

## Model Performance Patterns

### Thinking Mode Effects
- **Productive Efficiency (Athey-Bagwell):** TE models dominate (0.59 vs 0.29 for TD)
- **Allocative Efficiency (Spulber):** Mixed — TD models sometimes better (Q3-14B TD: 0.74 vs TE: 0.28)
- **Pricing (Salop):** No clear TE/TD advantage
- **Coordination (Green-Porter):** Minimal differences

### Top Strategic Performers

#### Athey-Bagwell (Productive Efficiency)
1. Q3-14B TE: 0.595 (3P), 0.497 (5P)
2. Q3-235B Inst: 0.595 (3P), 0.497 (5P)
3. Qwen3-30B-A3B TD: 0.594 (3P), 0.483 (5P)

#### Spulber (Allocative Efficiency)
1. L4-Scout: 0.74 (3P), 0.75 (5P) — improves!
2. L4-Maverick: 0.65 (3P), 0.73 (5P) — improves!
3. Q3-14B TD: 0.70 (3P), 0.74 (5P) — improves!

#### Green-Porter (Low Reversion = Good)
1. Q3-14B TD: 0.127 (3P), 0.125 (5P)
2. Q3-235B Inst: 0.127 (3P), 0.125 (5P)
3. Multiple tied at 0.127 (3P)

#### Salop (Market Price)
1. L3.3-70B: 13.89 (3P), 13.10 (5P)
2. L3.1-70B: 13.80 (3P), 13.28 (5P)
3. Q3-32B TD: 13.66 (3P), 13.02 (5P)

### Weakest Performers

#### Athey-Bagwell
- Q3-14B TD: 0.289 (3P) — 2× worse than TE counterpart
- Random: 0.428 (3P), 0.300 (5P)

#### Spulber
- L3.3-70B: 0.35 (3P), 0.22 (5P)
- L3.1-8B: 0.40 (3P), 0.22 (5P)

#### Salop
- L3.1-8B: 11.32 (3P) — significantly underprices market

## Insights by Game Structure

### Athey-Bagwell (Capacity Constraints)
- **Clear TE advantage** in managing production capacity
- **Most dramatic 3P→5P decline** (efficiency drops ~10–16%)
- Suggests: Coordination becomes harder with more players

### Spulber (Search & Matching)
- **More players improve overall market efficiency**
- Individual profits fall, but social welfare improves
- L4-Scout and L4-Maverick excel at matching
- **Paradox:** Best matchers (Scout/Maverick) don't earn highest profits

### Green-Porter (Collusion under Uncertainty)
- **More players → LESS punishment**
- All models cluster around 12.5–13.5% reversion rate
- Low variance suggests models use similar strategies
- Counter-intuitive: Easier to sustain cooperation with more players?

### Salop (Product Differentiation)
- **Prices stable regardless of competition**
- Differentiation strategy maintains pricing power
- L3.1-8B struggles with pricing (underprices by 15–20%)
- High variance in 3P (σ = 0.12–1.66) suggests diverse strategies

## Relationship to Profit Performance

### Productive Efficiency (Athey-Bagwell)
- **High correlation with profit:** TE models with 0.59 efficiency earn 4300+ profit
- **Low efficiency = low profit:** Q3-14B TD (0.29 efficiency) earns 2720 profit

### Allocative Efficiency (Spulber)
- **WEAK correlation with profit:** L4-Scout has 0.75 efficiency but only 53 profit
- **Disconnect:** Good for market ≠ good for individual

### Reversion Frequency (Green-Porter)
- **NO clear correlation:** Models with similar reversion rates have vastly different profits
- Suggests: Demand shocks dominate strategy effects

### Market Price (Salop)
- **NEGATIVE correlation:** Highest prices (13.89) associate with negative profits (-67)
- Suggests: Overpricing reduces demand, lowers profit

## Data Quality Observations

### Identical Values
- Many models have identical values in Athey-Bagwell (0.595/0.497)
- Suggests: TE models converge to same production strategy
- Or: Possible data aggregation artifact

### Standard Deviations
- **Salop prices:** High variance (σ up to 1.66)
- **Efficiency metrics:** Moderate variance (σ = 0.11–0.50)
- **Reversion frequency:** Low variance (σ = 0.023–0.027) — highly stable

## Implications for RQ1

### Strategic Competence
- **Not all models understand game structure:**
  - L3.1-8B underprices in Salop (11.32 vs ~13.5 average)
  - L3.3-70B has poor allocative efficiency (0.22)

### Thinking Mode Effects
- **Strong in capacity/production games** (Athey-Bagwell)
- **Weak in coordination games** (Green-Porter)
- **Mixed in matching games** (Spulber)

### Competition Complexity
1. **Simple price competition (Salop):** Stable
2. **Production planning (Athey-Bagwell):** Degraded
3. **Matching (Spulber):** Improved (paradoxically)
4. **Collusion (Green-Porter):** Slightly improved

### Market Efficiency vs. Individual Profit
- **Spulber:** Social welfare improves, individual welfare declines
- **Athey-Bagwell:** Both decline
- **Green-Porter:** Collusion stability improves, but demand shocks dominate profits
- **Salop:** Prices stable, differentiation varies

## Related Files
- `T_perf_avg_profit.csv` — Profit outcomes for same games
- `T_perf_win_rate.csv` — Binary win outcomes
- `T_magic_*.csv` — Behavioral profiles (MAgIC metrics)
- `SYNTHESIS_RQ1_Competitive_Performance.md` — Full synthesis
