import json
import sys
import os
import random
from pathlib import Path
from openai import OpenAI

# --- CONFIGURATION ---
API_KEY = "nay79qYyoAEJ34wJesbL69WGZkOJtdB3"  # Replace with actual key
BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL = "Qwen/Qwen3-14B"  # Replace with desired model
OUTPUT_FILE = "reasoning_results_round50.json"

# --- HELPER: HISTORY GENERATORS ---

def generate_green_porter_history(current_round=50):
    """Generates 49 rounds of history for Green Porter."""
    history_lines = []
    prices = []
    
    # Simulation: Stable -> Price War (Round 40) -> Recovery
    for r in range(1, current_round):
        if (40 <= r <= 43) or r == 49:
            state = "Price War"
            price = 21.00 + random.uniform(-1, 1)
        elif r == current_round:
            state = "Price War"
        else:
            state = "Collusive"
            price = 65.00 + random.uniform(0, 1.5)
        
        history_lines.append(f"Period {r} | {state} | ${price:.2f}")
        prices.append(price)
        
    return "\n".join(history_lines), prices

def generate_athey_history(current_round=50, num_opponents=4):
    """Generates 49 rounds of detailed report history for Athey Bagwell."""
    my_reports = []
    others_reports = []
    
    for r in range(1, current_round):
        my_type = "high" if random.random() > 0.3 else "low"
        my_reports.append(f"Period {r}: {my_type}")
        
        ops = []
        for i in range(1, num_opponents + 1):
            op_type = "high" if random.random() > 0.3 else "low"
            ops.append(f"P{i}: {op_type}")
        others_reports.append(f"Period {r}: {', '.join(ops)}")
        
    return "; ".join(my_reports), "; ".join(others_reports)

# --- DATA LOADING ---

def load_game_config(game_name):
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        return config.get('game_configs', {}).get(game_name, {}).get('baseline', {})
    except FileNotFoundError:
        return {}

def get_filled_context(game_name):
    """Creates context for Round 50 with 5 Players."""
    cfg = load_game_config(game_name)
    
    context = {
        "player_id": "challenger", 
        "current_round": 50,
        "discount_factor": 0.9,
        **cfg 
    }

    if game_name == 'salop':
        # Standard Salop (Circular City) configuration
        num_players = 5
        circumference = 1.0
        
        context.update({
            "number_of_players": num_players,
            "circumference": circumference,
            "distance_to_neighbor": circumference / num_players, # Equals 0.2
            "market_size": context.get("market_size", 3600),     # Default to 3600 per config
            "transport_cost": 1.5,
            "reservation_price": 30,
            # 'max_brand_utility' and 'outside_good_surplus' are implicit in reservation_price 
            # but kept for completeness if you revert prompts.
            "max_brand_utility": 40,
            "outside_good_surplus": 10,
            "marginal_cost": 8,
            "fixed_cost": 100
        })

    elif game_name == 'spulber':
        context.update({
            "number_of_competitors": 4, 
            "your_cost": 8,
            "rival_cost_distribution": "normal",
            "rival_cost_mean": 10,
            "rival_cost_std": 2.0,
            "demand_intercept": 100,
            "demand_slope": 1
        })

    elif game_name == 'green_porter':
        history_str, price_list = generate_green_porter_history(50)
        
        # 5 Players Constants
        base_demand = 120
        slope = 1
        mc = 20
        num_players = 5
        q_cournot = 16.66
        q_collusive = 10.0

        # Calculations
        total_q_war = q_cournot * num_players
        expected_price_war_price = max(0, base_demand - (slope * total_q_war)) 
        expected_price_war_profit = (expected_price_war_price - mc) * q_cournot 

        total_q_defect = ((num_players - 1) * q_collusive) + q_cournot 
        defect_price = max(0, base_demand - (slope * total_q_defect)) 
        immediate_defect_profit = (defect_price - mc) * q_cournot 

        context.update({
            "number_of_players": num_players,
            "current_market_state": "Collusive", 
            "price_history": price_list,
            "price_history_length": 49,
            "formatted_history_table": history_str,
            "collusive_quantity": q_collusive,
            "cournot_quantity": q_cournot,
            "trigger_price": 60,
            "punishment_duration": 3,
            "marginal_cost": mc,
            "base_demand": base_demand,
            "demand_slope": slope,
            "demand_shock_mean": 0,
            "demand_shock_std": 7,
            "demand_shock_distribution": "normal",
            "expected_price_war_price": f"{expected_price_war_price:.2f}",
            "expected_price_war_profit": f"{expected_price_war_profit:.2f}",
            "immediate_defect_profit": f"{immediate_defect_profit:.2f}"
        })

    elif game_name == 'athey_bagwell':
        my_hist, others_hist = generate_athey_history(50, num_opponents=4)
        context.update({
            "number_of_players": 5,
            "your_cost_type": "low",
            "your_reports_history_detailed": my_hist,
            "all_other_reports_history_detailed": others_hist,
            "persistence_probability": 0.7,
            "market_size": 100,
            "market_price": 30
        })
        
    return context

# --- MAIN EXECUTION ---
games = ['salop']
all_results = {}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"🚀 Starting Round 50 (5-Player) reasoning tests...")
print(f"🤖 Model: {MODEL}\n")

for game in games:
    print(f"\n{'='*40}")
    print(f"🎮 {game.upper()}")
    print(f"{'='*40}")
    
    # 1. Load Template
    prompt_path = Path(f'prompts/{game}.md')
    if not prompt_path.exists():
        print(f"❌ Template missing: {prompt_path}")
        continue
    with open(prompt_path, 'r') as f:
        template = f.read()

    # 2. Fill Context
    try:
        context = get_filled_context(game)
        filled_prompt = template.format(**context)
    except KeyError as e:
        print(f"❌ Error filling variables: Missing {e}")
        continue

    # 3. PRINT INPUT TEXT (Requested)
    print(f"📝 INPUT PROMPT:\n{'-'*20}\n{filled_prompt}\n{'-'*20}\n")

    # 4. Call API
    print("⏳ Generating response...")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": filled_prompt}],
            stream=False,
            temperature=0.0,
            #reasoning_effort="high"  # Exclude reasoning tokens
        )
        
        # 5. Metrics & Calculation
        full_content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        total_out_tokens = response.usage.completion_tokens

        # --- CALCULATE REASONING TOKENS (OUTPUT - 20) ---
        # Assuming the final "answer" is roughly 20 tokens
        reasoning_tokens = max(0, total_out_tokens - 20)
        answer_tokens = total_out_tokens - reasoning_tokens # Should be roughly 20

        print(f"✅ Finished.")
        print(f"📊 METRICS:")
        print(f"   Input Tokens:      {input_tokens}")
        print(f"   Total Output:      {total_out_tokens}")
        print(f"   Reasoning Tokens:  {reasoning_tokens} (Total - 20)")
        
        all_results[game] = {
            "round": 50,
            "players": 5,
            "metrics": {
                "input_tokens": input_tokens,
                "total_output_tokens": total_out_tokens,
                "reasoning_tokens_calc": reasoning_tokens,
            },
            "input_prompt": filled_prompt,
            "full_response": full_content
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        all_results[game] = {"error": str(e)}

# --- SAVE ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n💾 Results saved to {OUTPUT_FILE}")