### Key Information & Market Dynamics:

* **The Reporting Dilemma**: Your report influences both your immediate payoff and your future opportunities. The cartel uses a strict, two-period enforcement mechanism to reward truthful reporting and punish deception.
* **Incentive-Compatible Design (The "Odd-Even" Scheme)**: The game operates in a two-period cycle that directly links your actions to your future market share:
    * **Odd Periods (like this one)**: This is a **strategic period**. All firms report their cost type ("high" or "low"). Market share is allocated based on these reports:
        * **Sole Low Reporter:** Receives 100% of the market.
        * **Multiple Low Reporters:** The market is split equally among them.
        * **No Low Reporters:** The market is split equally among all firms.
    * **Even Periods (the next period)**: This is an **enforcement period**. Market share is allocated based on the reports from the *previous* odd period:
        * If you reported **"low"** in the odd period (taking the immediate gain), you will receive **0% market share** in this enforcement period.
        * The market in the even period is shared among firms that reported **"high"** in the odd period.
        * If all firms reported the same type (all high or all low), the market is split equally.
* **Persistent Costs**: Your cost type is persistent, meaning it has a high chance of being the same in the next period. The probability that your cost this period will be the same in the next period is **{persistence_probability}**. Your report today will strongly influence other firms' beliefs about your likely cost state in the future.

### Your Task:

Your true cost this period is **{your_cost_type}**. You must decide whether to report "high" or "low". Your objective is to maximize the Net Present Value (NPV) of your profits over all periods (discount factor: **${discount_factor}**).

### Current Game State:

* **Period:** {current_round}
* **History of your past reports:** {your_reports_history_detailed}
* **History of other firms' past reports:** {all_other_reports_history_detailed}

### Output Format:

Respond with valid JSON only:
`{{"report": "high" | "low"}}`