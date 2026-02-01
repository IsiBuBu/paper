You are one of **{number_of_players}** firms competing in a circular market with a circumference of **{circumference}** and a total of **{market_size}** customers. Your nearest competitors are located at a distance of **{distance_to_neighbor}** on either side.

### The Physics of Your Market Share:
Your market share depends entirely on how your price ($P$) compares to your neighbors' price ($P_{{rival}}$). You face two distinct demand regimes:

1.  **Monopoly Boundary:** Customers are willing to travel to you only if their total cost (Price + Transport) is less than their reservation price ($R={reservation_price}$).
    * **Formula:** Maximum reach in one direction is $x_{{max}} = (R - P) / {transport_cost}$.
    **Strategic Implication:** If you charge a very high price, you shrink to a local monopoly and lose distant customers to the "Outside Good."

2.  **Competitive Boundary (The Critical Formula):**
    If your price is low enough to compete with your neighbors, your market share is NOT fixed. It is dynamic. The boundary between you and a neighbor occurs where a customer is indifferent:
    $$P + {transport_cost} \cdot x = P_{{rival}} + {transport_cost} \cdot ({distance_to_neighbor} - x)$$
    
    Solving for $x$ (your reach on one side), your total demand $Q$ (covering both left and right sides) becomes:
    $$Q = {market_size} \cdot \left( {distance_to_neighbor} + \frac{{P_{{rival}} - P}}{{transport_cost}} \right)$$

### The Strategic Calculus:
You must set a single price $P$ to maximize your **Profit**:
$$Profit = (P - {marginal_cost}) \cdot Q - {fixed_cost}$$

**Crucial Consideration:**
You do not know $P_{{rival}}$ for certain, but you must assume your rivals are rational profit-maximizers who face the exact same incentives as you. 
* If you set $P$ too high, $(P_{{rival}} - P)$ becomes negative, and you lose market share rapidly. 
* If you set $P$ too low, you gain share but sacrifice margin.

### Task:
Find the **Best Response Price**. This is the price where, assuming your rivals also play optimally (Nash Equilibrium), you cannot increase your profit by changing your price.

### Output Format:
You MUST respond with valid JSON first and only. Do not include any explanation or text:
`{{"price": <number>}}`