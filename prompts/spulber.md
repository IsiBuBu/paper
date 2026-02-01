You are one of **{number_of_players}** firms in a winner-take-all Bertrand competition. This is a game of incomplete information.

### The Strategic Context:
* **Your Position:** Your marginal cost is **${your_cost}**.
* **The Unknown:** You do not know your rivals' exact costs. You only know their costs are drawn independently from a **{rival_cost_distribution}** distribution (Mean=**${rival_cost_mean}**, SD=**${rival_cost_std}**).
* **The Market:** If you satisfy the winning condition, `Quantity = {demand_intercept} - Your_Price`.

### The Game Mechanics (Crucial):
1.  **Rational Rivals:** Your rivals are not pricing at cost, nor are they pricing randomly. They are rational profit-maximizers.
2.  **Winning Condition:** You win **only** if your price is strictly lower than the prices set by all **{number_of_competitors}** rivals.
3.  **Endogenous Risk:** Your probability of winning is **not fixed**. It is highly sensitive to your price.
    * If you price aggressively (low), you squeeze your margin but maximize your chance of undercutting rational rivals.
    * If you price passively (high), you expand your margin but risk that *at least one* rival draws a favorable cost and undercuts you.

### The Strategic Calculus:
You must estimate the likely pricing behavior of your rivals and determine the single price that maximizes your **Expected Profit**.
$$E[\pi] = P(Win) \times (Price - {your_cost}) \times Quantity$$

### Task:
Find the optimal price that balances your profit margin against the probability of undercutting your rivals.

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"price": <number>}}`