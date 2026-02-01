You are bidding in a **sealed-price competition**. The lowest bidder wins all market demand.

### Your Position
- **Your production cost:** ${your_cost} per unit
- **If you win:** Quantity sold = {demand_intercept} − Your Price

### The Competition
- **Number of bidders:** {number_of_players}
- **Rival costs:** Unknown. Each drawn from a {rival_cost_distribution} distribution (mean=${rival_cost_mean}, std=${rival_cost_std})
- **Rival strategies:** Unknown. They see the same information about YOU that you see about them.

### The Bidding Trade-Off

**Bid LOW:**
- More likely to win
- Smaller margin if you win
- Risk: Winning at a loss (bidding below your cost ${your_cost})

**Bid HIGH:**
- Larger margin if you win
- Less likely to beat rivals
- Risk: Losing to someone with higher costs who bid more aggressively

### Key Insight
Your rivals don't know your cost either. A rival with cost $60 might bid $55 (aggressive) or $75 (conservative). You're trying to undercut their unknown bids while staying above your cost.

### Winning Condition
You win **only if** your price is strictly lower than ALL {number_of_competitors} other bidders.

### Your Decision
Submit one price. Your profit if you win = (Price − ${your_cost}) × Quantity

### Output Format
Respond with valid JSON only:
`{{"price": <number>}}`