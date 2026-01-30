You are in a cartel with **{number_of_players}** firms. The cartel uses an "Odd-Even" allocation scheme to ensure productive efficiency (the lowest cost firm serves).

### The Scheme:
* **Odd Periods:** Firms report costs.
    * **Report "Low":** You claim the market *today*. You serve **{market_size}** units but are **barred** from the market in the next period.
    * **Report "High":** You yield priority. You get **0** share now (unless everyone yields), but are guaranteed access to the market in the next period.
* **Even Periods:** Market share is allocated to those who yielded (reported "High") in the previous period.

### Economic Context:
* **Market Price:** **${market_price}**.
* **Cost Structure:** Low Cost (**${low_cost}**) vs. High Cost (**${high_cost}**).
* **Cost Persistence:** Your cost type is sticky. The probability of your cost staying the same in the next period is **{persistence_probability}**, regardless of whether it is Low or High.

### The Strategic Calculus:
Your true cost this period is **{your_cost_type}** (${your_true_cost_value}). You must calculate if it is better to seize profit now or invest in the future (Discount Factor: **{discount_factor}**).

1.  **Report "Low" (Claim):**
    * **Consequence:** You serve the market today.
    * **Payoff:** You earn an immediate profit of **~${profit_claim_today}**.
    * **Risk:** You are barred tomorrow. If your cost becomes Low tomorrow, you miss a massive opportunity.

2.  **Report "High" (Yield):**
    * **Consequence:** You yield priority to Low-cost firms.
    * **Payoff:** You likely earn **0** today, but you gain guaranteed access to the next period. 
    * **Future Value:** Based on your cost probabilities, the Expected NPV of this future access is **~${expected_future_value}**.

### Current Game State:
* **Period:** {current_round}
* **History of your past reports:** {your_reports_history_detailed}

### Task:
Which report maximizes your firm's expected NPV given your current cost type?

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"report": "high" | "low"}}`