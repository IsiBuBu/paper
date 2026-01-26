import asyncio
import logging
import json
import sys
import os
from pathlib import Path

# 1. Setup Python Path
sys.path.append(str(Path(__file__).parent))

from config.config import load_config, get_all_game_configs
from agents import create_agent
from games import create_game

# Configure Logging to STDERR so STDOUT stays clean for JSON output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("GameTest")

# Define critical parameters to check
CRITICAL_PARAMS = {
    "salop": ["distance_to_neighbor", "transport_cost", "reservation_price", "market_size"],
    "spulber": ["rival_cost_mean", "rival_cost_std", "demand_intercept", "number_of_players"],
    "green_porter": ["trigger_price", "punishment_duration", "cournot_quantity", "collusive_quantity"],
    "athey_bagwell": ["persistence_probability", "market_price", "discount_factor"]
}

async def run_test():
    """
    Runs a smoke test for ALL games/models and outputs a JSON report to stdout.
    Now includes the INPUT PROMPT in the output.
    """
    
    # Data container for final JSON output
    test_report = {
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "results": []
    }

    try:
        full_config = load_config()
    except FileNotFoundError:
        logger.critical("❌ Could not load config/config.json.")
        return

    challenger_models = full_config['models']['challenger_models']
    all_game_names = list(full_config.get('game_configs', {}).keys())
    
    test_report["summary"]["total"] = len(all_game_names) * len(challenger_models)
    
    for game_name in all_game_names:
        logger.info(f"🧪 TESTING GAME: {game_name.upper()}")
        
        game_configs = get_all_game_configs(game_name)
        if not game_configs:
            logger.error(f"No configuration found for {game_name}!")
            continue
            
        config = game_configs[0]
        
        # Initialize Game & Mock State
        try:
            game = create_game(game_name)
            game_state = game.initialize_game_state(config, simulation_id=0)
            
            # Mock Data Injection
            if game_name == 'athey_bagwell':
                game_state['current_period'] = 1
                game_state['cost_sequences'] = {'challenger': ['low'] * 50}
                game_state['report_history'] = {'challenger': [], 'defender_1': []}
            elif game_name == 'spulber':
                game_state['player_private_costs'] = {'challenger': config.constants.get('your_cost', 10)}
            elif game_name == 'green_porter':
                game_state['price_history'] = [55.0, 60.0]
                game_state['state_history'] = ['Collusive', 'Collusive']
            
        except Exception as e:
            logger.exception(f"❌ Failed to initialize game {game_name}")
            continue

        # Parameter Validation
        state_constants = game_state.get('constants', config.constants)
        missing_params = []
        for param in CRITICAL_PARAMS.get(game_name, []):
            if state_constants.get(param) is None and game_state.get(param) is None:
                if param != "expected_price_war_profit": # Skip derived
                    missing_params.append(param)
        
        if missing_params:
            logger.error(f"❌ Missing params in {game_name}: {missing_params}")
            # Log failure for all models in this game
            for model_name in challenger_models:
                test_report["results"].append({
                    "game": game_name,
                    "model": model_name,
                    "status": "config_error",
                    "error": f"Missing params: {missing_params}"
                })
            continue

        # Run Model Tests
        for model_name in challenger_models:
            logger.info(f"🤖 Model: {model_name}")
            
            result_entry = {
                "game": game_name,
                "model": model_name,
                "status": "unknown",
                "metrics": {},
                "input": {},  # <--- New Field
                "output": {}
            }

            try:
                agent = create_agent(model_name, "challenger", mock_mode=False)
                
                # Generate and capture Prompt
                prompt = game.generate_player_prompt("challenger", game_state, config)
                result_entry["input"]["prompt_text"] = prompt  # <--- Store Prompt
                
                call_id = f"test-{game_name}-{model_name.split('/')[-1]}"
                
                response = await agent.get_response(prompt, call_id, config)
                
                if response.success:
                    result_entry["status"] = "success"
                    test_report["summary"]["passed"] += 1
                    
                    result_entry["metrics"] = {
                        "response_time_sec": round(response.response_time, 2),
                        "tokens_used": response.tokens_used,
                        "thinking_tokens": response.thinking_tokens,
                        "reasoning_char_count": response.reasoning_char_count
                    }
                    
                    result_entry["output"] = {
                        "json_content": response.content,
                        "reasoning_content": response.reasoning_content if response.reasoning_content else None
                    }
                else:
                    result_entry["status"] = "api_error"
                    result_entry["error"] = response.error
                    test_report["summary"]["failed"] += 1

            except Exception as e:
                logger.exception(f"💥 Critical error on {model_name}")
                result_entry["status"] = "critical_error"
                result_entry["error"] = str(e)
                test_report["summary"]["failed"] += 1
            
            test_report["results"].append(result_entry)

    # OUTPUT FINAL JSON TO STDOUT
    print(json.dumps(test_report, indent=2))

if __name__ == "__main__":
    if not os.getenv("OPENAI_BASE_URL") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_BASE_URL/API_KEY not set. Defaulting to localhost:8000")
        
    asyncio.run(run_test())