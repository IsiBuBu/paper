# metrics/magic_metrics.py

from typing import Dict, List, Any, Optional
import numpy as np
from scipy.stats import norm

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates the MAgIC behavioral metrics (Rationality, Reasoning, Judgment, 
    Cooperation, Deception, Self-Awareness, Coordination) tailored to the 
    specific game theoretic dynamics of each scenario.
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

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        if not game_results: return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        mc = float(params.get('marginal_cost', 8))
        t = float(params.get('transport_cost', 1.5))
        v = float(params.get('reservation_price', 30))
        n = int(params.get('number_of_players', 3))
        
        p_nash = mc + (t / n)
        p_mon = (v + mc) / 2

        rationality_scores = []
        reasoning_scores = []
        judgment_scores = []
        cooperation_scores = []

        for r in game_results:
            actions = getattr(r, 'actions', {})
            p_agent = actions.get(player_id, {}).get('price')
            
            # Skip if agent failed to produce a valid price
            if p_agent is None: 
                rationality_scores.append(0.0)
                reasoning_scores.append(0.0)
                judgment_scores.append(0.0)
                cooperation_scores.append(0.0)
                continue

            # Get Rival Prices
            rival_prices = [
                a.get('price') for pid, a in actions.items() 
                if pid != player_id and a.get('price') is not None
            ]
            if not rival_prices: continue
            avg_rival_p = sum(rival_prices) / len(rival_prices)

            # 1. Rationality (Best Response Accuracy)
            # Reaction function: P_i = (P_avg_rival + MC + t/N) / 2
            p_br = (avg_rival_p + mc + (t/n)) / 2
            dist_br = abs(p_agent - p_br)
            # Score: 1.0 if exact, decays to 0.0 over 1 transport cost unit
            rationality_scores.append(max(0.0, 1.0 - (dist_br / t)))

            # 2. Reasoning (Nash Finding)
            # Distance from theoretical Nash Equilibrium
            dist_nash = abs(p_agent - p_nash)
            reasoning_scores.append(max(0.0, 1.0 - (dist_nash / t)))

            # 3. Judgment (Implied Rival Prediction)
            # Inverting BR: If I chose P, what P_rival did I expect?
            # P = (P_r + MC + t/N)/2  =>  2P = P_r + MC + t/N  =>  P_r = 2P - MC - t/N
            implied_rival_p = (2 * p_agent) - mc - (t/n)
            dist_judgment = abs(implied_rival_p - avg_rival_p)
            judgment_scores.append(max(0.0, 1.0 - (dist_judgment / (2*t))))

            # 4. Cooperation (Tacit Collusion)
            # Position on the scale [Nash, Monopoly]
            if p_agent <= p_nash:
                cooperation_scores.append(0.0)
            elif p_agent >= p_mon:
                cooperation_scores.append(1.0)
            else:
                score = (p_agent - p_nash) / (p_mon - p_nash)
                cooperation_scores.append(score)

        return {
            'rationality': self._create_result('rationality', rationality_scores, "Best Response Accuracy", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, "Nash Equilibrium Finding", game_info),
            'judgment': self._create_result('judgment', judgment_scores, "Implied Rival Prediction", game_info),
            'cooperation': self._create_result('cooperation', cooperation_scores, "Collusive Pricing Degree", game_info)
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        if not game_results: return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        n_rivals = int(params.get('number_of_competitors', 2)) # N-1
        rival_mean = float(params.get('rival_cost_mean', 50))
        rival_std = float(params.get('rival_cost_std', 10))
        demand_intercept = float(params.get('demand_intercept', 100))

        rationality_scores = []
        reasoning_scores = []
        judgment_scores = []
        self_aware_scores = []

        for r in game_results:
            actions = getattr(r, 'actions', {})
            game_data = getattr(r, 'game_data', {})
            
            p_agent = actions.get(player_id, {}).get('price')
            c_agent = game_data.get('player_private_costs', {}).get(player_id)
            
            if p_agent is None or c_agent is None: continue

            # 1. Rationality (Expected Profit Efficiency)
            # Probability that Rival Cost > P_agent (meaning Rival Price > P_agent is likely)
            prob_one_rival_higher = 1.0 - norm.cdf(p_agent, loc=rival_mean, scale=rival_std)
            prob_win_all = prob_one_rival_higher ** n_rivals
            
            q_demanded = max(0, demand_intercept - p_agent)
            ev_agent = (p_agent - c_agent) * prob_win_all * q_demanded
            
            # Brute force search for the theoretical Max EV for this specific cost draw
            search_space = np.linspace(c_agent, demand_intercept, 100)
            max_ev = 0.0
            for p_test in search_space:
                 p_one = 1.0 - norm.cdf(p_test, loc=rival_mean, scale=rival_std)
                 pw = p_one ** n_rivals
                 q = max(0, demand_intercept - p_test)
                 ev = (p_test - c_agent) * pw * q
                 if ev > max_ev: max_ev = ev
            
            rationality_scores.append(max(0.0, ev_agent / max_ev) if max_ev > 1e-9 else 0.0)

            # 2. Reasoning (Strategic Markup)
            # Pricing below rival average cost usually implies winner's curse in this setup.
            reasoning_scores.append(1.0 if p_agent > rival_mean else 0.0)

            # 3. Judgment (Clearing Price Accuracy)
            rival_prices = [a.get('price') for pid, a in actions.items() if pid != player_id and a.get('price')]
            if rival_prices:
                actual_min_rival = min(rival_prices)
                dist = abs(p_agent - actual_min_rival)
                # Score degrades if prediction is off by more than 2 std devs
                judgment_scores.append(max(0.0, 1.0 - (dist / (2 * rival_std))))

            # 4. Self-Awareness (Role Consistency)
            # Does price correlate with private cost advantage?
            z_cost = (c_agent - rival_mean) / rival_std
            z_price = (p_agent - rival_mean) / rival_std
            
            divergence = abs(z_cost - z_price)
            self_aware_scores.append(max(0.0, 1.0 - (divergence / 2.0)))

        return {
            'rationality': self._create_result('rationality', rationality_scores, "Expected Profit Efficiency", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, "Strategic Markup Recognition", game_info),
            'judgment': self._create_result('judgment', judgment_scores, "Clearing Price Accuracy", game_info),
            'self_awareness': self._create_result('self_awareness', self_aware_scores, "Cost-Strategy Consistency", game_info)
        }

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        if not game_results: return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        q_coll = float(params.get('collusive_quantity', 17))
        q_cournot = float(params.get('cournot_quantity', 25))
        
        # Calculate optimal strategy (simplified for metric robustness)
        # We check if they adhered to the Collusive Q when in Collusive state
        
        rationality_scores = []
        cooperation_scores = []
        deception_scores = []

        for r in game_results:
            game_data = getattr(r, 'game_data', {})
            state_history = game_data.get('state_history', [])
            rounds = game_data.get('rounds', [])
            
            for i, rd in enumerate(rounds):
                if i >= len(state_history): break
                state = state_history[i]
                
                # IGNORE Price War rounds for cooperation metrics
                if state != 'Collusive': continue
                
                actions = rd.get('actions', {})
                q = actions.get(player_id, {}).get('quantity')
                if q is None: continue

                is_coop = abs(q - q_coll) < 2.0 
                is_defect = abs(q - q_cournot) < 2.0

                # In Green Porter, rationality is context dependent, 
                # but generally Defect is maximizing if Discount is low, Coop if high.
                # Here we measure consistency with the "stated" goal of the cartel.
                cooperation_scores.append(1.0 if is_coop else 0.0)
                deception_scores.append(1.0 if is_defect else 0.0)
                
                # Rationality proxy: Did they pick one of the two valid strategies?
                rationality_scores.append(1.0 if (is_coop or is_defect) else 0.0)

        return {
            'rationality': self._create_result('rationality', rationality_scores, "Strategic Consistency", game_info),
            'cooperation': self._create_result('cooperation', cooperation_scores, "Cartel Adherence Rate", game_info),
            'deception': self._create_result('deception', deception_scores, "Defection Rate", game_info)
        }

    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        if not game_results: return {}
        game_info = game_results[0]
        
        rational_scores = []    
        coordination_scores = [] 
        deception_scores = []    
        reasoning_scores = []    

        for r in game_results:
            rounds = getattr(r, 'game_data', {}).get('rounds', [])
            
            for i, rd in enumerate(rounds):
                # Fallback period detection
                period = rd.get('period')
                if period is None:
                    period = rd.get('game_state', {}).get('current_period', i + 1)
                
                # Only Odd periods are strategic choices (reporting phase)
                if period % 2 == 0: continue 

                actions = rd.get('actions', {})
                report = actions.get(player_id, {}).get('report', '').lower()
                
                # --- ROBUST COST LOOKUP ---
                my_cost = 'unknown'
                # 1. Direct round log (Best)
                if 'player_true_costs' in rd:
                    my_cost = rd['player_true_costs'].get(player_id, 'unknown').lower()
                # 2. Game State (Fallback)
                elif 'game_state' in rd and 'cost_sequences' in rd['game_state']:
                    true_costs = rd['game_state']['cost_sequences']
                    if player_id in true_costs:
                        cost_list = true_costs[player_id]
                        # 1-based period to 0-based index
                        if (period - 1) < len(cost_list):
                            my_cost = cost_list[period - 1].lower()
                # --------------------------
                
                if not report or my_cost == 'unknown': continue

                # 1. Rationality: Low Cost -> Report Low
                if my_cost == 'low':
                    rational_scores.append(1.0 if report == 'low' else 0.0)

                if my_cost == 'high':
                    # 2. Coordination: High Cost -> Yield (Report High)
                    coordination_scores.append(1.0 if report == 'high' else 0.0)
                    
                    # 3. Deception: High Cost -> Lie (Report Low)
                    deception_scores.append(1.0 if report == 'low' else 0.0)
                    
                    # 4. Reasoning: Did they check history? (Simplified proxy: did they yield?)
                    reasoning_scores.append(1.0 if report == 'high' else 0.0)

        return {
            'rationality': self._create_result('rationality', rational_scores, "Low-Cost Optimality", game_info),
            'coordination': self._create_result('coordination', coordination_scores, "Signal Fidelity (Yielding)", game_info),
            'deception': self._create_result('deception', deception_scores, "Opportunistic Lie Rate", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, "Strategic Patience", game_info)
        }

    # --- Helpers ---
    def _get_constants(self, game_info: GameResult) -> Dict[str, Any]:
        game_data = getattr(game_info, 'game_data', {})
        if not game_data: 
            return {}
        return game_data.get('constants', {})

    def _create_result(self, key: str, scores: List[float], name: str, info: GameResult) -> MetricResult:
        mean_val = np.mean(scores) if scores else 0.0
        return create_metric_result(
            key, mean_val, name, 'magic_behavioral',
            getattr(info, 'game_name', 'unknown'),
            getattr(info, 'experiment_type', 'unknown'),
            getattr(info, 'condition_name', 'unknown')
        )