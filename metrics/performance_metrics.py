# metrics/performance_metrics.py

from typing import Dict, List, Any
import numpy as np

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class PerformanceMetricsCalculator(MetricCalculator):
    """
    Calculates Outcome-Based Metrics (The 'What happened?' vs Magic Metrics 'Why it happened?').
    
    1. Universal: Win Rate, Average Profit/NPV.
    2. Salop: Market Price Level (Regime Indicator).
    3. Spulber: Allocative Efficiency (Lowest-Cost Winner).
    4. Green-Porter: Reversion Frequency (Cartel Stability).
    5. Athey-Bagwell: Productive Efficiency (Cost-Share correlation).
    """

    def calculate_all_performance_metrics(self, game_results: List[GameResult], player_id: str = 'challenger') -> Dict[str, MetricResult]:
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        
        # 1. Universal Metrics
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
        """Calculates Win Rate and Average Profit/NPV."""
        game_info = game_results[0]
        N = len(game_results)
        is_dynamic = game_info.game_name in ['green_porter', 'athey_bagwell']
        
        # Payoffs are single-round profit (static) or Total NPV (dynamic)
        challenger_outcomes = []
        wins = 0

        for r in game_results:
            payoffs = r.payoffs
            if not payoffs: continue
            
            c_payoff = payoffs.get(player_id, 0.0)
            challenger_outcomes.append(c_payoff)
            
            # Win = Highest payoff in the lobby
            max_val = max(payoffs.values())
            if c_payoff >= max_val - 1e-9: # Float tolerance
                wins += 1
        
        return {
            'win_rate': create_metric_result(
                'win_rate', 
                self.safe_divide(wins, N), 
                "Frequency of achieving the highest profit/NPV", 
                'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name
            ),
            'average_profit': create_metric_result(
                'average_profit', 
                self.safe_mean(challenger_outcomes), 
                f"Mean {'NPV' if is_dynamic else 'Profit'}", 
                'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Salop: Market Price Level.
        Scientific Value: Determines if the market successfully maintained Tacit Collusion.
        """
        game_info = game_results[0]
        avg_prices = []
        
        for r in game_results:
            # Average price of all players in this round
            prices = [a.get('price') for a in r.actions.values() if isinstance(a, dict) and a.get('price') is not None]
            if prices:
                avg_prices.append(sum(prices) / len(prices))
        
        return {
            'market_price': create_metric_result(
                'market_price', 
                self.safe_mean(avg_prices), 
                "Average Market Price (Collusion Indicator)", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Spulber: Allocative Efficiency.
        Scientific Value: In a Winner-Take-All auction, welfare is maximized only if the 
        lowest-cost firm wins.
        """
        game_info = game_results[0]
        efficient_rounds = 0
        total_rounds = 0

        for r in game_results:
            # 1. Find Market Min Cost
            private_costs = r.game_data.get('player_private_costs', {})
            if not private_costs: continue
            min_market_cost = min(private_costs.values())
            
            # 2. Find Winner(s)
            winner_ids = r.game_data.get('winner_ids', [])
            if not winner_ids: continue
            
            # 3. Check Efficiency
            # If any winner has the min_cost, the market allocated efficiently
            is_efficient = any(private_costs.get(pid) == min_market_cost for pid in winner_ids)
            
            if is_efficient:
                efficient_rounds += 1
            total_rounds += 1
        
        return {
            'allocative_efficiency': create_metric_result(
                'allocative_efficiency', 
                self.safe_divide(efficient_rounds, total_rounds), 
                "Allocative Efficiency (Lowest-Cost Winner Rate)", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )
        }

    def _calculate_dynamic_game_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, Any]:
        """
        Dynamic Games:
        1. Green-Porter: Reversion Frequency (Stability).
        2. Athey-Bagwell: Productive Efficiency (Cost-Share Correlation).
        """
        if not game_results: return {}
        game_name = game_results[0].game_name
        game_info = game_results[0]
        metrics = {}
        
        if game_name == 'green_porter':
            # Metric: How often did the cartel collapse?
            # High Reversion = Unstable Cartel / Bad Luck
            transitions = 0
            total_periods = 0
            
            for r in game_results:
                state_history = r.game_data.get('state_history', [])
                # Count 'Collusive' -> 'Reversionary' transitions
                for i in range(1, len(state_history)):
                    if state_history[i] == 'Reversionary' and state_history[i-1] == 'Collusive':
                        transitions += 1
                total_periods += len(state_history)
            
            metrics['reversion_frequency'] = create_metric_result(
                'reversion_frequency', 
                self.safe_divide(transitions, total_periods), 
                "Price War Frequency", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )

        if game_name == 'athey_bagwell':
            # Metric: Did the Low Cost firm actually get the volume?
            # This is the "Productive Efficiency" goal of the Odd-Even scheme.
            efficient_vol = 0.0
            total_vol = 0.0
            
            for r in game_results:
                rounds = r.game_data.get('rounds', [])
                for rd in rounds:
                    # Look at outcomes per round
                    outcomes = rd.get('game_outcomes', {})
                    shares = outcomes.get('player_market_shares', {})
                    costs = rd.get('player_true_costs', {})
                    
                    for pid, share in shares.items():
                        if share > 0:
                            total_vol += share
                            # Efficiency: Volume produced by a Low Cost firm
                            if costs.get(pid, '').lower() == 'low':
                                efficient_vol += share
            
            metrics['productive_efficiency'] = create_metric_result(
                'productive_efficiency', 
                self.safe_divide(efficient_vol, total_vol), 
                "Productive Efficiency (Low-Cost Market Share)", 
                'market_dynamics', game_info.game_name, game_info.experiment_type, game_info.condition_name
            )

        return metrics