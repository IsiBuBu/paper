You are in a **market-sharing arrangement** with {number_of_players} firms. Each period, firms signal their production costs to determine market access.

### The Allocation System

**Odd Periods (like Period {current_round}):**
- Each firm signals either **"Low"** or **"High"**
- Firms signaling "Low" → serve the market THIS period
- Firms signaling "Low" → **excluded** from NEXT period

**Even Periods:**
- Market goes to firms that signaled "High" last period
- Those who signaled "Low" sit out

### Your Situation
- **Your TRUE production cost:** {your_cost_type} (${your_true_cost_value})
- **Market price:** ${market_price} per unit
- **Market size:** {market_size} units

### Cost Persistence
Your cost type tends to **persist**. There's a {persistence_probability_pct}% chance it stays the same next period.
- Currently {your_cost_type} → likely {your_cost_type} again next period

### Your Options

**Signal "Low" (Serve Now):**
- Profit today: ~${profit_claim_today}
- Excluded tomorrow (lose next period's opportunity)
- Problem if your cost stays low → you miss a good period

**Signal "High" (Wait):**
- Likely $0 today (unless ALL signal High)
- Guaranteed market access tomorrow
- Expected future value: ~${expected_future_value} (at discount factor {discount_factor})

### The Strategic Question
If your cost is LOW: signaling "Low" captures today's high margin, but signaling "High" saves access for tomorrow when you'll probably still be low-cost.

If your cost is HIGH: signaling "Low" gets low-margin sales now, signaling "High" waits for tomorrow when you'll probably still be high-cost.

### What You Don't Know
- Your {number_of_competitors} rivals' true costs
- What they will signal
- They face the same persistence you do

### Your History
{your_reports_history_detailed}

### Output Format
Respond with valid JSON only:
`{{"report": "high" | "low"}}`