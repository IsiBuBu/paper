You are a firm in a **circular market** competing on price. Your location is fixed; customers choose the lowest total cost (price + travel).

### Market Structure
- **Competitors:** {number_of_players} firms, evenly spaced around the circle
- **Distance to neighbors:** {distance_to_neighbor:.3f} units on each side
- **Total customers:** {market_size}
- **Travel cost:** ${transport_cost} per unit distance customers travel
- **Your production cost:** ${marginal_cost} per unit (plus ${fixed_cost} fixed)

### How Customers Decide
A customer considers their total cost: **Your Price + Travel Cost to Reach You**

They buy from whoever offers the lowest total cost—but only if that total is below their maximum willingness to pay (${reservation_price}). Otherwise, they don't buy at all.

### The Pricing Trade-Off
- **Price HIGH:** Better margin per sale, but customers may travel to neighbors or not buy
- **Price LOW:** Attract more customers, but thinner margins

### What You Don't Know
Your neighbors set their prices **simultaneously**. You cannot see their choices before deciding. They face the same trade-offs you do, but may:
- Price aggressively to steal customers
- Price high hoping everyone does
- Miscalculate

### Your Task
Set a price that balances margin against market share, given uncertainty about your neighbors' pricing.

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"price": <number>}}`