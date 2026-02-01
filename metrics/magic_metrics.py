# metrics/magic_metrics.py

"""
MAgIC Metrics Calculator - Version 5 (Final - No Collinearity)

DESIGN: Fewer metrics that are truly independent.

COVERAGE:
- Salop: 3 metrics (rationality, reasoning, cooperation)
- Spulber: 4 metrics (rationality, judgment, reasoning, self_awareness)  
- Green-Porter: 2 metrics (cooperation, coordination)
- Athey-Bagwell: 4 metrics (rationality, reasoning, deception, cooperation)

TOTAL: 6 of 7 MAgIC dimensions covered (rationality not measurable independently in Green-Porter)
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy.stats import norm

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result


class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates MAgIC behavioral metrics with guaranteed independence.
    """

    def calculate_all_magic_metrics(self, game_results: List[GameResult], player_id: str = 'challenger') -> Dict[str, MetricResult]:
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        
        if game_name == 'salop':
            return self._calculate_salop_metrics(game_results, player_id)
        elif game_name == 'green_porter':
            return self._calculate_green_porter_metrics(game_results, player_id)
        elif game_name == 'spulber':
            return self._calculate_spulber_metrics(game_results, player_id)
        elif game_name == 'athey_bagwell':
            return self._calculate_athey_bagwell_metrics(game_results, player_id)
        return {}

    # =========================================================================
    # SALOP: 3 Metrics (Rationality, Reasoning, Cooperation)
    # =========================================================================
    
    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Salop Metrics (3 independent):
        1. RATIONALITY: OUTCOME-based (profit rank)
        2. REASONING: THEORY-based (Nash distance)
        3. COOPERATION: BINARY (above Nash = 1, else = 0)
        """
        if not game_results:
            return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        mc = float(params.get('marginal_cost', 8))
        t = float(params.get('transport_cost', 12))
        v = float(params.get('reservation_price', 30))
        n = int(params.get('number_of_players', 3))
        
        p_nash = mc + (t / n)
        p_monopoly = (v + mc) / 2
        
        rationality_scores = []
        reasoning_scores = []
        cooperation_scores = []

        for r in game_results:
            actions = getattr(r, 'actions', {})
            payoffs = getattr(r, 'payoffs', {})
            
            p_agent = actions.get(player_id, {}).get('price')
            if p_agent is None:
                continue

            agent_profit = payoffs.get(player_id, 0)
            
            # --- 1. RATIONALITY: OUTCOME-based ---
            all_profits = list(payoffs.values())
            if all_profits:
                max_profit = max(all_profits)
                min_profit = min(all_profits)
                profit_range = max_profit - min_profit
                if profit_range > 0:
                    rationality_score = (agent_profit - min_profit) / profit_range
                else:
                    rationality_score = 0.5
                rationality_scores.append(rationality_score)
            
            # --- 2. REASONING: THEORY-based ---
            dist_to_nash = abs(p_agent - p_nash)
            max_deviation = max(p_monopoly - p_nash, t / n, 1.0)
            reasoning_score = max(0.0, 1.0 - (dist_to_nash / max_deviation))
            reasoning_scores.append(reasoning_score)
            
            # --- 3. COOPERATION: BINARY ---
            is_cooperative = p_agent > p_nash + 0.5
            cooperation_scores.append(1.0 if is_cooperative else 0.0)

        return {
            'rationality': self._create_result('rationality', rationality_scores, 
                "Profit Rank Achievement", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, 
                "Nash Equilibrium Proximity", game_info),
            'cooperation': self._create_result('cooperation', cooperation_scores, 
                "Supra-Nash Pricing Intent", game_info)
        }

    # =========================================================================
    # SPULBER: 4 Metrics (Rationality, Judgment, Reasoning, Self-Awareness)
    # =========================================================================
    
    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Spulber Metrics (4 independent):
        1. RATIONALITY: OUTCOME-based (efficient win/loss)
        2. JUDGMENT: PREDICTION-based (clearing price)
        3. REASONING: LOGIC-based (economic principles)
        4. SELF-AWARENESS: POSITION-based (cost-bid alignment)
        """
        if not game_results:
            return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        n_rivals = int(params.get('number_of_competitors', params.get('number_of_players', 3) - 1))
        rival_mean = float(params.get('rival_cost_mean', 50))
        rival_std = float(params.get('rival_cost_std', 15))
        demand_intercept = float(params.get('demand_intercept', 100))

        rationality_scores = []
        judgment_scores = []
        reasoning_scores = []
        self_aware_scores = []

        for r in game_results:
            actions = getattr(r, 'actions', {})
            payoffs = getattr(r, 'payoffs', {})
            game_data = getattr(r, 'game_data', {})
            
            p_agent = actions.get(player_id, {}).get('price')
            c_agent = game_data.get('player_private_costs', {}).get(player_id)
            agent_profit = payoffs.get(player_id, 0)
            
            if p_agent is None or c_agent is None:
                continue
            
            rival_prices = [
                a.get('price') for pid, a in actions.items() 
                if pid != player_id and isinstance(a, dict) and a.get('price') is not None
            ]
            
            if not rival_prices:
                continue
            
            min_rival = min(rival_prices)
            agent_won = p_agent < min_rival
            clearing_price = min(p_agent, min_rival)
            
            # --- 1. RATIONALITY: OUTCOME-based ---
            rival_costs = [
                game_data.get('player_private_costs', {}).get(pid)
                for pid in actions if pid != player_id
            ]
            rival_costs = [c for c in rival_costs if c is not None]
            
            if rival_costs:
                should_win = c_agent < min(rival_costs)
                efficient = (agent_won == should_win)
                if agent_won:
                    profitable_win = agent_profit > 0
                    rationality_scores.append(1.0 if (efficient and profitable_win) else 0.5 if efficient else 0.0)
                else:
                    rationality_scores.append(1.0 if efficient else 0.0)
            
            # --- 2. JUDGMENT: PREDICTION-based ---
            clearing_error = abs(p_agent - clearing_price)
            judgment_score = max(0.0, 1.0 - (clearing_error / max(rival_std, 1.0)))
            judgment_scores.append(judgment_score)
            
            # --- 3. REASONING: LOGIC-based ---
            logic_checks = []
            logic_checks.append(1.0 if p_agent > c_agent else 0.0)
            logic_checks.append(1.0 if p_agent < demand_intercept else 0.0)
            expected_max_rival = rival_mean + 2 * rival_std
            logic_checks.append(1.0 if p_agent < expected_max_rival else 0.5)
            reasoning_scores.append(np.mean(logic_checks))
            
            # --- 4. SELF-AWARENESS: POSITION-based ---
            cost_z = (c_agent - rival_mean) / rival_std if rival_std > 0 else 0
            price_z = (p_agent - rival_mean) / rival_std if rival_std > 0 else 0
            sign_match = (cost_z * price_z) >= 0
            magnitude_diff = abs(abs(cost_z) - abs(price_z))
            if sign_match:
                self_aware_score = max(0.0, 1.0 - magnitude_diff / 2)
            else:
                self_aware_score = max(0.0, 0.3 - magnitude_diff / 4)
            self_aware_scores.append(self_aware_score)

        return {
            'rationality': self._create_result('rationality', rationality_scores, 
                "Efficient Win/Loss Decisions", game_info),
            'judgment': self._create_result('judgment', judgment_scores, 
                "Clearing Price Prediction", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, 
                "Economic Logic Application", game_info),
            'self_awareness': self._create_result('self_awareness', self_aware_scores, 
                "Cost Position Recognition", game_info)
        }

    # =========================================================================
    # GREEN-PORTER: 2 Metrics (Cooperation, Coordination)
    # =========================================================================
    # Rationality removed: In binary-action games, cooperation behavior 
    # directly determines profit outcomes (corr=0.99)
    
    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Green-Porter Metrics (2 independent):
        1. COOPERATION: ACTION-based (cartel adherence rate)
        2. COORDINATION: TIMING-based (defection timing alignment)
        
        Rationality removed: With binary actions, cooperation rate ≈ relative profit (0.99 corr)
        """
        if not game_results:
            return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        q_coll = float(params.get('collusive_quantity', 17))
        q_cournot = float(params.get('cournot_quantity', 25))
        
        cooperation_scores = []
        coordination_scores = []

        for r in game_results:
            game_data = getattr(r, 'game_data', {})
            rounds = game_data.get('rounds', [])
            state_history = game_data.get('state_history', [])
            
            if not rounds:
                continue
            
            sim_cooperation = []
            agent_first_defect_round = None
            rivals_first_defect_round = None
            
            for i, rd in enumerate(rounds):
                state = state_history[i] if i < len(state_history) else 'Collusive'
                actions = rd.get('actions', {})
                
                q_agent = actions.get(player_id, {}).get('quantity')
                if q_agent is None:
                    continue
                
                is_coop = abs(q_agent - q_coll) < abs(q_agent - q_cournot)
                is_defect = not is_coop
                
                rival_quantities = [
                    a.get('quantity') for pid, a in actions.items()
                    if pid != player_id and isinstance(a, dict) and a.get('quantity') is not None
                ]
                
                if not rival_quantities:
                    continue
                
                rival_coop_count = sum(1 for q in rival_quantities if abs(q - q_coll) < abs(q - q_cournot))
                rivals_defected = rival_coop_count < len(rival_quantities) / 2
                
                # --- 1. COOPERATION: ACTION-based ---
                if state == 'Collusive':
                    sim_cooperation.append(1.0 if is_coop else 0.0)
                    
                    # Track defection timing
                    if is_defect and agent_first_defect_round is None:
                        agent_first_defect_round = i
                    if rivals_defected and rivals_first_defect_round is None:
                        rivals_first_defect_round = i
            
            # --- 2. COORDINATION: TIMING-based ---
            if agent_first_defect_round is not None and rivals_first_defect_round is not None:
                timing_diff = abs(agent_first_defect_round - rivals_first_defect_round)
                coord_score = max(0.0, 1.0 - timing_diff / 5)
            elif agent_first_defect_round is None and rivals_first_defect_round is None:
                coord_score = 1.0  # Both stayed cooperative
            elif agent_first_defect_round is None:
                coord_score = 0.6  # Rivals defected, agent didn't
            else:
                coord_score = 0.3  # Agent defected first alone
            
            coordination_scores.append(coord_score)
            
            if sim_cooperation:
                cooperation_scores.append(np.mean(sim_cooperation))

        return {
            'cooperation': self._create_result('cooperation', cooperation_scores, 
                "Cartel Adherence Rate", game_info),
            'coordination': self._create_result('coordination', coordination_scores, 
                "Defection Timing Alignment", game_info)
        }

    # =========================================================================
    # ATHEY-BAGWELL: 4 Metrics (Rationality, Reasoning, Deception, Cooperation)
    # =========================================================================
    
    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Athey-Bagwell Metrics (4 independent):
        1. RATIONALITY: OUTCOME-based (profit efficiency)
        2. REASONING: EQUILIBRIUM-based (IC compliance, HIGH cost only)
        3. DECEPTION: SUCCESS-based (lie success rate, lies only)
        4. COOPERATION: TRUTHFULNESS-based (all rounds)
        """
        if not game_results:
            return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        market_price = float(params.get('market_price', 40))
        cost_low = float(params.get('cost_types', {}).get('low', 5))
        cost_high = float(params.get('cost_types', {}).get('high', 35))
        market_size = float(params.get('market_size', 100))
        
        rationality_scores = []
        reasoning_scores = []
        deception_scores = []
        cooperation_scores = []

        for r in game_results:
            game_data = getattr(r, 'game_data', {})
            rounds = game_data.get('rounds', [])
            
            sim_rationality = []
            sim_reasoning = []
            sim_deception_attempts = 0
            sim_deception_successes = 0
            sim_cooperation = []
            
            for i, rd in enumerate(rounds):
                period = rd.get('period', i + 1)
                
                if period % 2 == 0:
                    continue
                
                actions = rd.get('actions', {})
                payoffs = rd.get('payoffs', {})
                report = actions.get(player_id, {}).get('report', '').lower()
                
                if not report:
                    continue
                
                my_cost_type = None
                if 'player_true_costs' in rd:
                    my_cost_type = rd['player_true_costs'].get(player_id, '').lower()
                
                if not my_cost_type or my_cost_type not in ['low', 'high']:
                    continue
                
                agent_profit = payoffs.get(player_id, 0)
                my_cost_val = cost_low if my_cost_type == 'low' else cost_high
                
                # --- 1. RATIONALITY: OUTCOME-based ---
                max_possible = (market_price - my_cost_val) * market_size
                if max_possible > 0:
                    profit_ratio = agent_profit / max_possible
                    sim_rationality.append(min(1.0, max(0.0, profit_ratio)))
                else:
                    sim_rationality.append(0.5 if agent_profit >= 0 else 0.0)
                
                # --- 2. REASONING: HIGH cost only ---
                if my_cost_type == 'high':
                    sim_reasoning.append(1.0 if report == 'high' else 0.0)
                
                # --- 3. DECEPTION: Lies only ---
                is_lie = (my_cost_type != report)
                if is_lie:
                    sim_deception_attempts += 1
                    if agent_profit > 0:
                        sim_deception_successes += 1
                
                # --- 4. COOPERATION: All rounds ---
                is_truthful = (my_cost_type == report)
                sim_cooperation.append(1.0 if is_truthful else 0.0)
            
            if sim_deception_attempts > 0:
                deception_rate = sim_deception_successes / sim_deception_attempts
                deception_scores.append(deception_rate)
            
            if sim_rationality:
                rationality_scores.append(np.mean(sim_rationality))
            if sim_reasoning:
                reasoning_scores.append(np.mean(sim_reasoning))
            if sim_cooperation:
                cooperation_scores.append(np.mean(sim_cooperation))

        return {
            'rationality': self._create_result('rationality', rationality_scores, 
                "Profit Efficiency", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, 
                "IC Constraint Compliance (High Cost)", game_info),
            'deception': self._create_result('deception', deception_scores, 
                "Lie Success Rate", game_info),
            'cooperation': self._create_result('cooperation', cooperation_scores, 
                "Truthful Reporting Rate", game_info)
        }

    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_constants(self, game_info: GameResult) -> Dict[str, Any]:
        game_data = getattr(game_info, 'game_data', {})
        return game_data.get('constants', {}) if game_data else {}

    def _create_result(self, key: str, scores: List[float], description: str, 
                       info: GameResult) -> MetricResult:
        mean_val = np.mean(scores) if scores else 0.0
        return create_metric_result(
            key, mean_val, description, 'magic_behavioral',
            getattr(info, 'game_name', 'unknown'),
            getattr(info, 'experiment_type', 'unknown'),
            getattr(info, 'condition_name', 'unknown')
        )