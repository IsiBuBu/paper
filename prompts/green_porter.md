You operate in a **quantity-setting oligopoly** with {number_of_players} firms. Market price depends on total industry output.

### Production Options
Choose ONE output level this period:

| Option | Your Output | Market Impact |
|--------|-------------|---------------|
| **Restrain** | {collusive_quantity} units | Lower supply → higher price |
| **Expand** | {cournot_quantity} units | More output → more units sold |

### Market Mechanics
- **Base demand:** {base_demand}
- **Your cost:** ${marginal_cost} per unit
- **Price volatility:** Random demand shocks (std dev: {demand_shock_std})

### The Price Monitoring System
The industry monitors a **trigger price of ${trigger_price}**.
- If market price falls below this trigger → **Low-output phase** for {punishment_duration} periods
- During this phase: expected price ~${expected_price_war_price}, profit ~${expected_price_war_profit}/period

### Expected Profits (This Period)
- **Restrain** (if others also restrain): ~${expected_cooperate_profit}
- **Expand** (if others restrain): ~${immediate_defect_profit}
- Expanding increases the risk of triggering a price collapse

### What You Don't Know
- Your {number_of_competitors} rivals choose **simultaneously**
- They may also be considering expansion
- Multiple expansions → more severe price drop
- **Rival strategies are uncertain**

### Game Progress
- **Period:** {current_round} of {time_horizon}
- **Rounds remaining:** {rounds_remaining}
- **Status:** {current_market_state}

### End-Game Effect
With {rounds_remaining} rounds left, a {punishment_duration}-period punishment covers {effective_punishment_periods} of your remaining periods. The threat of future punishment diminishes as the game nears its end.

### Recent History
{formatted_history_table}

### Output Format
Respond with valid JSON only:
`{{"action": "Restrain" | "Expand"}}`