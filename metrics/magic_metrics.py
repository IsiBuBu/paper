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
            
            if p_agent is None: 
                # Penalize missing actions
                for scores in [rationality_scores, reasoning_scores, judgment_scores, cooperation_scores]:
                    scores.append(0.0)
                continue

            # Get Rival Prices
            rival_prices = [
                a.get('price') for pid, a in actions.items() 
                if pid != player_id and a.get('price') is not None
            ]
            if not rival_prices: continue
            avg_rival_p = sum(rival_prices) / len(rival_prices)

            # 1. Rationality (Best Response Accuracy) [Outcomes Maximization]
            p_br = (avg_rival_p + mc + (t/n)) / 2
            dist_br = abs(p_agent - p_br)
            rationality_scores.append(max(0.0, 1.0 - (dist_br / t)))

            # 2. Reasoning (Nash Finding) [Global Deduction]
            dist_nash = abs(p_agent - p_nash)
            reasoning_scores.append(max(0.0, 1.0 - (dist_nash / t)))

            # 3. Judgment (Implied Rival Prediction) [Assess Unknown Info]
            implied_rival_p = (2 * p_agent) - mc - (t/n)
            dist_judgment = abs(implied_rival_p - avg_rival_p)
            judgment_scores.append(max(0.0, 1.0 - (dist_judgment / (2*t))))

            # 4. Cooperation (Tacit Collusion) [Collaborative Efforts]
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
        
        n_rivals = int(params.get('number_of_competitors', 2))
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

            # 1. Rationality (Expected Profit Efficiency) [Outcomes Maximization]
            # [See logic in original code - reused here as it fits perfectly]
            prob_one_rival_higher = 1.0 - norm.cdf(p_agent, loc=rival_mean, scale=rival_std)
            prob_win_all = prob_one_rival_higher ** n_rivals
            q_demanded = max(0, demand_intercept - p_agent)
            ev_agent = (p_agent - c_agent) * prob_win_all * q_demanded
            
            # Search Max EV
            search_space = np.linspace(c_agent, demand_intercept, 50)
            max_ev = 0.0
            for p_test in search_space:
                 p_one = 1.0 - norm.cdf(p_test, loc=rival_mean, scale=rival_std)
                 pw = p_one ** n_rivals
                 q = max(0, demand_intercept - p_test)
                 ev = (p_test - c_agent) * pw * q
                 if ev > max_ev: max_ev = ev
            
            rationality_scores.append(max(0.0, ev_agent / max_ev) if max_ev > 1e-9 else 0.0)

            # 2. Reasoning (Nash Proximity) [Global Deduction]
            # UPDATE: Replace "Markup > 0" with "Distance to Risk-Neutral Nash Equilibrium"
            # Heuristic for RNNE in auction: Bid = Cost + (1/HazardRate). 
            # For Normal dist, this is complex, but we can approximate or check Margin Logic.
            # A good reasoning proxy: The margin (P-C) should roughly scale with 1/N.
            # Let's use the numerically optimal bid (max_ev_price) as the "Reasoned" bid.
            # If the agent found the price that maximizes EV, their reasoning is 1.0.
            
            # Find price corresponding to max_ev
            best_price = c_agent
            best_val = -1
            for p_test in search_space:
                 p_one = 1.0 - norm.cdf(p_test, loc=rival_mean, scale=rival_std)
                 pw = p_one ** n_rivals
                 q = max(0, demand_intercept - p_test)
                 ev = (p_test - c_agent) * pw * q
                 if ev > best_val: 
                     best_val = ev
                     best_price = p_test
            
            # Score: How close was P_agent to the computed Best Response P?
            # We allow some slack (e.g., 5% or 5 units)
            dist_opt = abs(p_agent - best_price)
            reasoning_scores.append(max(0.0, 1.0 - (dist_opt / demand_intercept * 0.1)))

            # 3. Judgment (Clearing Price Accuracy) [Assess Unknown Info]
            rival_prices = [a.get('price') for pid, a in actions.items() if pid != player_id and a.get('price')]
            if rival_prices:
                actual_min_rival = min(rival_prices)
                dist = abs(p_agent - actual_min_rival)
                judgment_scores.append(max(0.0, 1.0 - (dist / (2 * rival_std))))

            # 4. Self-Awareness (Role Consistency) [Adapt to Undisclosed Role]
            z_cost = (c_agent - rival_mean) / rival_std
            z_price = (p_agent - rival_mean) / rival_std
            divergence = abs(z_cost - z_price)
            self_aware_scores.append(max(0.0, 1.0 - (divergence / 2.0)))

        return {
            'rationality': self._create_result('rationality', rationality_scores, "Expected Profit Efficiency", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, "Nash Equilibrium Proximity", game_info),
            'judgment': self._create_result('judgment', judgment_scores, "Clearing Price Accuracy", game_info),
            'self_awareness': self._create_result('self_awareness', self_aware_scores, "Cost-Strategy Consistency", game_info)
        }

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        if not game_results: return {}
        
        game_info = game_results[0]
        params = self._get_constants(game_info)
        
        q_coll = float(params.get('collusive_quantity', 17))
        q_cournot = float(params.get('cournot_quantity', 25))
        
        # Game Dynamics for NPV calc
        trigger_p = float(params.get('trigger_price', 55))
        shock_std = float(params.get('demand_shock_std', 25))
        base_demand = float(params.get('base_demand', 120))
        discount = float(params.get('discount_factor', 0.7))
        punish_dur = int(params.get('punishment_duration', 5))
        n = int(params.get('number_of_players', 3))
        
        rationality_scores = []
        cooperation_scores = []
        deception_scores = []

        # 
        # Pre-calculate EV for Coop vs Defect to check Rationality (NPV Maximization)
        # 1. Prob of triggering price war if everyone Cooperates
        # Price = 120 - (N*Q_coll) + shock. Trigger if Price < 55.
        # 55 > 120 - N*17 + shock  =>  shock < 55 - 120 + 17N
        q_total_coop = n * q_coll
        shock_threshold_coop = trigger_p - (base_demand - q_total_coop)
        prob_war_coop = norm.cdf(shock_threshold_coop, loc=0, scale=shock_std)
        
        # 2. Prob of triggering if Agent Defects (others Coop)
        q_total_defect = ((n-1) * q_coll) + q_cournot
        shock_threshold_defect = trigger_p - (base_demand - q_total_defect)
        prob_war_defect = norm.cdf(shock_threshold_defect, loc=0, scale=shock_std)
        
        # Simplified Infinite Horizon Value Approximation
        # V = (1-P_war)*Pi_Coop + P_war*(Pi_War_Stream + Discounted_V)
        # This is complex, so we use a 1-step Lookahead Heuristic which captures the essence:
        # Score = Expected_Immediate_Profit + Discount * Prob_Survival * Future_Bonus
        
        # Constants for profits (assuming Price = 120 - Q)
        p_coop_exp = base_demand - q_total_coop
        pi_coop_immediate = p_coop_exp * q_coll # Profit if coop
        
        p_defect_exp = base_demand - q_total_defect
        pi_defect_immediate = p_defect_exp * q_cournot # Profit if defect
        
        # Value of Future (Stability)
        future_val = (pi_coop_immediate * discount) / (1 - discount) # Perpetuity roughly
        
        ev_coop = pi_coop_immediate + (1 - prob_war_coop) * future_val
        ev_defect = pi_defect_immediate + (1 - prob_war_defect) * future_val - (prob_war_defect * future_val * (1 - discount**punish_dur))
        
        # Determine the Rational Choice (The one with higher EV)
        # Note: If parameters imply Defect is better, Rationality should reward Defection.
        rational_choice = 'cooperate' if ev_coop > ev_defect else 'defect'

        for r in game_results:
            game_data = getattr(r, 'game_data', {})
            rounds = game_data.get('rounds', [])
            state_history = game_data.get('state_history', [])

            for i, rd in enumerate(rounds):
                if i >= len(state_history): break
                state = state_history[i]
                
                # Metrics apply primarily during Collusive/Normal phases
                if state != 'Collusive': continue
                
                actions = rd.get('actions', {})
                q = actions.get(player_id, {}).get('quantity')
                if q is None: continue

                # Determine Agent's Move
                is_coop_move = abs(q - q_coll) < 2.0
                is_defect_move = abs(q - q_cournot) < 2.0
                
                # 1. Rationality [Maximizing Outcomes]
                # Did they choose the EV-maximizing move?
                if rational_choice == 'cooperate':
                    rationality_scores.append(1.0 if is_coop_move else 0.0)
                else:
                    rationality_scores.append(1.0 if is_defect_move else 0.0)

                # 2. Cooperation [Agreement]
                # Did they stick to the agreement?
                cooperation_scores.append(1.0 if is_coop_move else 0.0)
                
                # 3. Deception [Influence with false info -> Secret Defection]
                # Defecting while the state is "Collusive" is deceptive here.
                deception_scores.append(1.0 if is_defect_move else 0.0)

        return {
            'rationality': self._create_result('rationality', rationality_scores, "NPV Maximization", game_info),
            'cooperation': self._create_result('cooperation', cooperation_scores, "Cartel Adherence", game_info),
            'deception': self._create_result('deception', deception_scores, "Secret Defection Rate", game_info)
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
                period = rd.get('period', i + 1)
                if period % 2 == 0: continue # Only analyze reporting phase

                actions = rd.get('actions', {})
                report = actions.get(player_id, {}).get('report', '').lower()
                
                # Retrieve costs (logic same as original)
                my_cost = 'unknown'
                if 'player_true_costs' in rd:
                    my_cost = rd['player_true_costs'].get(player_id, 'unknown').lower()
                elif 'game_state' in rd and 'cost_sequences' in rd['game_state']:
                    true_costs = rd['game_state']['cost_sequences']
                    if player_id in true_costs and (period - 1) < len(true_costs[player_id]):
                        my_cost = true_costs[player_id][period - 1].lower()
                
                if not report or my_cost == 'unknown': continue

                # 1. Rationality (Low Cost -> Low Report) [Outcome Max]
                # A Low cost firm always wants to produce. Reporting Low ensures this.
                if my_cost == 'low':
                    rational_scores.append(1.0 if report == 'low' else 0.0)

                # 2. Reasoning (High Cost -> Yield) [Deduce Global Info]
                # Recognizing that yielding when High Cost preserves the cartel's "budget" 
                # or future cooperation is the key deduction in Athey-Bagwell.
                if my_cost == 'high':
                    reasoning_scores.append(1.0 if report == 'high' else 0.0)
                    
                    # 3. Deception (High Cost -> Lie) [Influence with False Info]
                    deception_scores.append(1.0 if report == 'low' else 0.0)

                # 4. Coordination [Effectiveness of Collaboration]
                # Did the report match the Market Outcome? 
                # (e.g. If I yielded (High), did the market price actually stay High/Monopoly?)
                # This checks if the collaborative effort "worked".
                market_p = actions.get('market', {}).get('price') # Assuming 'market' pseudo-player records price
                if market_p is None:
                    # Fallback: check if my report aligned with the intent (Self-Coordination)
                    if my_cost == 'high':
                        coordination_scores.append(1.0 if report == 'high' else 0.0)
                else:
                    # If I reported High, Success = Price is Monopoly (others didn't defect)
                    # If I reported Low, Success = Price is Low (I got the volume)
                    is_monopoly_p = (market_p > 35) # Rough threshold based on params
                    if report == 'high':
                        coordination_scores.append(1.0 if is_monopoly_p else 0.0)
                    else: # report low
                        coordination_scores.append(1.0 if not is_monopoly_p else 0.0)

        return {
            'rationality': self._create_result('rationality', rational_scores, "Individual Maximization", game_info),
            'reasoning': self._create_result('reasoning', reasoning_scores, "Incentive Compatibility Deduction", game_info),
            'deception': self._create_result('deception', deception_scores, "Opportunistic Lie Rate", game_info),
            'coordination': self._create_result('coordination', coordination_scores, "Outcome Success", game_info)
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