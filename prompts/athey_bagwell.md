* **The Scheme (Odd-Even Allocation)**: You are in a cartel with **{number_of_players}** firms.The cartel operates on a strict two-period cycle designed to ensure **Productive Efficiency**: the firm with the lowest cost should serve the market.
    * **Odd Periods (Strategic)**: Firms announce their cost types.
        * **Low Cost Claim:** If you report "low", you are claiming the right to serve the market today. You will receive 100% market share (or split it if others also claim).
        * **High Cost Claim:** If you report "high", you yield the market to others today.
    * **Even Periods (Balancing)**: Market share is allocated to "pay back" the firms that yielded in the previous period.
        * **The Cost of Claiming:** If you reported "low" in the odd period, you are **barred** from the market in the even period (0% share). This ensures you only claim the market when your cost is truly low.
        * **The Reward for Yielding:** If you reported "high" in the odd period, you share the market in the even period.

* **Persistent Costs**: Your cost type is persistent. If your cost is **Low** today, it is likely (Probability **{persistence_probability}**) to remain **Low** in the next period.
    * *Strategic Implication:* If you report "low" today, you lose the market tomorrow. If your cost stays Low tomorrow, that is a painful loss. You must weigh the **Immediate Profit** of serving the market today against the **Future Loss** of being barred tomorrow.

### Your Task:

Your true cost this period is **{your_cost_type}**. 
You must decide whether to report "high" or "low". 
Your objective is to maximize the Net Present Value (NPV) of your profits (discount factor: **${discount_factor}**).

* *Note:* Reporting "low" when your cost is Low is **Truthful**, not deceptive. It is the intended behavior of the scheme, provided the immediate profit outweighs the future restriction.

### Current Game State:

* **Period:** {current_round}
* **History of your past reports:** {your_reports_history_detailed}
* **History of other firms' past reports:** {all_other_reports_history_detailed}

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"report": "high" | "low"}}`