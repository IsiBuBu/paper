You are in a cartel with **{number_of_players}** firms. Your only goal is to maximize the Net Present Value (NPV) of your own profits.

### Cartel Rules:
* **The Noise:** The market price is volatile due to random demand shocks (std dev: **{demand_shock_std}**). 
* **The Deterrent:** If the price falls below **${trigger_price}**, the cartel collapses into a **Price War** for **{punishment_duration}** rounds.
* **The War:** During a Price War, the expected market price drops to **${expected_price_war_price}**, and your profits drop significantly to **~${expected_price_war_profit}** per round.

### The Strategic Calculus:
You must weigh the stability of the cartel against the temptation of immediate liquidity. Future profits are discounted by **{discount_factor}**.

1. **Cooperate** (Produce **{collusive_quantity}**):
   - **Reward:** You expect a profit of **~${expected_cooperate_profit}** today.
   - **Risk:** Even if you cooperate, a negative demand shock could hit the **${trigger_price}** trigger, starting a war through no fault of your own.

2. **Defect** (Produce **{cournot_quantity}**):
   - **Reward:** You capture an immediate profit bonus, totaling **~${immediate_defect_profit}**.
   - **Risk:** You significantly increase the mathematical probability of a market collapse, sacrificing **{punishment_duration}** rounds of future high-margin income.

### Current Game State:
* **Period:** {current_round}
* **Market Status:** {current_market_state}
* **History:**
{formatted_history_table}

### Task:
Which action maximizes your firm's expected NPV?

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"action": "Cooperate" | "Defect"}}`