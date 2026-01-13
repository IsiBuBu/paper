# metrics/performance_metrics.py

from typing import Dict, List, Any

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class PerformanceMetricsCalculator(MetricCalculator):
    """
    Calculates standard performance and market dynamics metrics.
    
    Includes:
    1. Universal Metrics: Win Rate, Average Profit/NPV.
    2. Market Dynamics: Game-specific indicators of market health (e.g., Efficiency, Stability).
    """

    def calculate_all_performance_metrics(self, game_results: List[GameResult], player_id: str = 'challenger') -> Dict[str, MetricResult]:
        """
        Calculates all applicable performance metrics for a given list of game results.
        This method acts as a dispatcher to the appropriate game-specific function.
        """
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        
        # 1. Universal Metrics (Win Rate, Profit)
        metrics = self._calculate_universal_metrics(game_results, player_id)

        # 2. Game-Specific Market Dynamics
        if game_name == 'salop':
            metrics.update(self._calculate_salop_metrics(game_results, player_id))
        elif game_name == 'spulber':
            metrics.update(self._calculate_spulber_metrics(game_results, player_id))
        elif game_name in ['green_porter', 'athey_bagwell']:
            metrics.update(self._calculate_dynamic_game_metrics(game_results, player_id))
        
        return metrics

    def _calculate_universal_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Calculates performance metrics that are common across all games.
        excludes profit volatility as requested.
        """
        game_info = game_results[0]
        N = len(game_results)
        
        is_dynamic = game_info.game_name in ['green_porter', 'athey_bagwell']
        
        # The 'payoffs' object correctly holds single-round profit for static games
        # and the final NPV for dynamic games.
        all_player_outcomes = [r.payoffs for r in game_results]
        challenger_outcomes = [r.payoffs.get(player_id, 0.0) for r in game_results]

        # --- Win Rate ---
        # A win is defined as achieving the highest profit/NPV in the simulation
        wins = 0
        for i, outcomes in enumerate(all_player_outcomes):
            if outcomes:
                max_val = max(outcomes.values())
                if challenger_outcomes[i] == max_val:
                    wins += 1
        
        win_rate = self.safe_divide(wins, N)
        
        # --- Average Profit / NPV ---
        avg_outcome = self.safe_mean(challenger_outcomes)
        
        profit_metric_name = "Average NPV" if is_dynamic else "Average Profit"
        
        return {
            'win_rate': create_metric_result(
                'win_rate', 
                win_rate, 
                "Frequency of achieving the highest profit/NPV", 
                'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name
            ),
            'average_profit': create_metric_result(
                'average_profit', 
                avg_outcome, 
                f"Mean of the challenger's {profit_metric_name}", 
                'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Calculates Salop Market Dynamics: Average Market Price.
        Used to diagnose the market regime (Competitive vs. Monopolistic).
        """
        game_info = game_results[0]
        avg_prices = []
        
        for r in game_results:
            # Get all prices in this round from all players
            prices = [a.get('price') for a in r.actions.values() if a.get('price') is not None]
            if prices:
                avg_prices.append(sum(prices) / len(prices))
        
        market_price = self.safe_mean(avg_prices)
        
        return {
            'market_price': create_metric_result(
                'market_price', 
                market_price, 
                "Average Price Level (Indicator of Competitive vs. Monopoly Regime)", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Calculates Spulber Market Dynamics: Allocative Efficiency.
        Measures if the winner was actually the lowest-cost firm.
        """
        game_info = game_results[0]
        efficiency_counts = 0
        total_rounds = 0

        for r in game_results:
            # 1. Identify the actual lowest cost in the market
            private_costs = r.game_data.get('player_private_costs', {})
            if not private_costs: continue
            
            min_cost = min(private_costs.values())
            
            # 2. Identify the winner's cost
            winner_ids = r.game_data.get('winner_ids', [])
            if not winner_ids: continue
            
            # If ANY winner had the min_cost, it's efficient (even if tied)
            is_efficient = any(private_costs.get(pid) == min_cost for pid in winner_ids)
            
            if is_efficient:
                efficiency_counts += 1
            total_rounds += 1
        
        allocative_efficiency = self.safe_divide(efficiency_counts, total_rounds)
        
        return {
            'allocative_efficiency': create_metric_result(
                'allocative_efficiency', 
                allocative_efficiency, 
                "Allocative Efficiency: Freq where winner was Lowest Cost firm", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_dynamic_game_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, Any]:
        """
        Calculates dynamic market metrics:
        1. Green & Porter: Reversion Frequency (Stability).
        2. Athey & Bagwell: Productive Efficiency (Low-Cost Production Share).
        """
        game_name = game_results[0].game_name
        game_info = game_results[0]
        metrics = {}
        
        if game_name == 'green_porter':
            # Reversion Frequency: The rate of system collapse into punishment
            reversion_counts = 0
            total_rounds = 0
            
            for r in game_results:
                state_history = r.game_data.get('state_history', [])
                # Count how many times we ENTERED reversion from Collusive
                switches = 0
                for i in range(1, len(state_history)):
                    if state_history[i] == 'Reversionary' and state_history[i-1] == 'Collusive':
                        switches += 1
                
                reversion_counts += switches
                total_rounds += len(state_history)
            
            # Probability of switch per period
            reversion_freq = self.safe_divide(reversion_counts, total_rounds)
            
            metrics['reversion_frequency'] = create_metric_result(
                'reversion_frequency', 
                reversion_freq, 
                "Reversion Frequency: Probability of switching to Punishment state", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )

        if game_name == 'athey_bagwell':
            # Productive Efficiency: Share of total production held by Low Cost firms
            # Ideally, Low Cost firms should have 100% of market share.
            efficient_production = 0.0
            total_production = 0.0
            
            for r in game_results:
                rounds = r.game_data.get('rounds', [])
                for rd in rounds:
                    market_shares = rd.get('game_outcomes', {}).get('player_market_shares', {})
                    true_costs = rd.get('player_true_costs', {})
                    
                    for pid, share in market_shares.items():
                        if share > 0:
                            total_production += share
                            if true_costs.get(pid) == 'low':
                                efficient_production += share
            
            prod_efficiency = self.safe_divide(efficient_production, total_production)
            
            metrics['productive_efficiency'] = create_metric_result(
                'productive_efficiency', 
                prod_efficiency, 
                "Productive Efficiency: Proportion of market share allocated to Low Cost firms", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )

        return metrics