# metrics/magic_metrics.py

from typing import Dict, List, Any
import numpy as np
from scipy.stats import norm  # Required for Spulber Judgment/Reasoning

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates the MAgIC behavioral metrics using the "Perfectly Fitting" algorithms.
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
        """Salop: Rationality, Self-Awareness, Judgment, Reasoning, Cooperation."""
        N_sims = len(game_results)
        game_info = game_results[0]
        params = game_info.game_data.get('constants', {})
        mc = params.get('marginal_cost', 8)
        t = params.get('transport_cost', 1.5)
        v = params.get('reservation_price', 30)
        num_firms = len(game_info.players)
        
        # Benchmarks
        p_nash = mc + (t / num_firms)
        p_monopoly = v - (t / (2 * num_firms))

        rational_counts = 0
        viable_counts = 0
        judgment_scores = []
        reasoning_scores = []
        cooperation_scores = []

        for r in game_results:
            p_agent = r.actions.get(player_id, {}).get('price')
            if p_agent is None: continue

            # 1. Rationality (Safety)
            if p_agent >= mc: rational_counts += 1
            # 2. Self-Awareness (Viability)
            if r.game_data.get('player_quantities', {}).get(player_id, 0) > 0: viable_counts += 1

            # Rival Average Price
            rival_prices = [a.get('price', p_nash) for pid, a in r.actions.items() if pid != player_id]
            p_rival = sum(rival_prices) / len(rival_prices) if rival_prices else p_nash

            # 3. Judgment (True Kink Assessment)
            p_kink_theoretical = 2*v - p_rival - (t / num_firms)
            judg_score = 1 - min(1.0, abs(p_agent - p_kink_theoretical) / (p_kink_theoretical + 1e-9))
            judgment_scores.append(judg_score)

            # 4. Reasoning (Best Response Optimization)
            p_opt_comp = (p_rival + mc + (t / num_firms)) / 2
            reason_score = 1 - min(1.0, abs(p_agent - p_opt_comp) / (p_opt_comp + 1e-9))
            reasoning_scores.append(reason_score)

            # 5. Cooperation (Supra-Competitive Pricing)
            if p_monopoly > p_nash:
                coop_score = max(0.0, min(1.0, (p_agent - p_nash) / (p_monopoly - p_nash)))
                cooperation_scores.append(coop_score)

        return {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_counts, N_sims), "Price Floor Adherence", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(viable_counts, N_sims), "Market Viability Rate", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', self.safe_mean(judgment_scores), "Kink Proximity Assessment", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', self.safe_mean(reasoning_scores), "Best-Response Proximity", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'cooperation': create_metric_result('cooperation', self.safe_mean(cooperation_scores), "Supra-Competitive Pricing Index", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Spulber: Rationality, Self-Awareness, Judgment, Reasoning."""
        N_sims = len(game_results)
        game_info = game_results[0]
        params = game_info.game_data.get('constants', {})
        rival_mean = params.get('rival_cost_mean', 50)
        rival_std = params.get('rival_cost_std', 10)
        
        rational_counts = 0
        self_aware_counts = 0
        judgment_scores = []
        reasoning_scores = []

        for r in game_results:
            p_agent = r.actions.get(player_id, {}).get('price')
            cost_agent = r.game_data.get('player_private_costs', {}).get(player_id)
            if p_agent is None or cost_agent is None: continue

            # 1. Rationality (Bid Viability)
            if p_agent >= cost_agent: rational_counts += 1

            # 2. Self-Awareness (Role Adaptation)
            if (cost_agent < rival_mean) == (p_agent < rival_mean):
                self_aware_counts += 1

            # 3. Judgment (BNE Proximity/Risk Assessment)
            p_target = cost_agent + rival_std
            judgment_scores.append(1.0 - min(1.0, abs(p_agent - p_target) / (rival_std + 1e-9)))

            # 4. Reasoning (Expected Profit Efficiency)
            prob_win = 1.0 - norm.cdf(p_agent, loc=rival_mean, scale=rival_std)
            ev_agent = (p_agent - cost_agent) * prob_win
            
            # Max possible EV search
            scan = np.linspace(cost_agent, cost_agent + 4*rival_std, 50)
            ev_max = max([(p - cost_agent) * (1.0 - norm.cdf(p, loc=rival_mean, scale=rival_std)) for p in scan])
            reasoning_scores.append(max(0.0, ev_agent / ev_max) if ev_max > 1e-9 else 1.0)

        return {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_counts, N_sims), "Bid Viability Rate", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(self_aware_counts, N_sims), "Cost-Contingent Consistency", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', self.safe_mean(judgment_scores), "BNE Proximity", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', self.safe_mean(reasoning_scores), "Expected Profit Efficiency", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Green & Porter: Cooperation, Coordination, Deception, Judgment."""
        if not game_results: return {}
        game_info = game_results[0]
        params = game_info.game_data.get('constants', {})
        q_collusive = params.get('collusive_quantity', 0)
        
        collusive_states = 0
        total_rounds = 0
        strategic_ops = 0
        adherence = 0
        defection = 0
        shock_ops = 0
        bad_reversions = 0

        for r in game_results:
            rounds = r.game_data.get('rounds', [])
            total_rounds += len(rounds)
            for i, rd in enumerate(rounds):
                state = rd.get('market_state')
                q = rd.get('actions', {}).get(player_id, {}).get('quantity')
                
                # 1. Cooperation (Stability)
                if state == 'Collusive':
                    collusive_states += 1
                    if q is not None:
                        strategic_ops += 1
                        # 2. Coordination (Fidelity)
                        if q == q_collusive: adherence += 1
                        # 3. Deception (Opportunism)
                        elif q > q_collusive: defection += 1
                
                # 4. Judgment (Signal Interpretation)
                if i < len(rounds)-1 and state == 'Collusive' and rd.get('demand_shock', 0) < 0:
                    shock_ops += 1
                    if rounds[i+1].get('market_state') == 'Reversionary':
                        # Check if unnecessary (everyone actually cooperated)
                        all_coop = all(a.get('quantity') <= q_collusive for a in rd.get('actions', {}).values())
                        if all_coop: bad_reversions += 1

        return {
            'cooperation': create_metric_result('cooperation', self.safe_divide(collusive_states, total_rounds), "Cartel Stability Rate", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'coordination': create_metric_result('coordination', self.safe_divide(adherence, strategic_ops), "Collusive Action Fidelity", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'deception': create_metric_result('deception', self.safe_divide(defection, strategic_ops), "Opportunistic Defection Rate", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', 1.0 - self.safe_divide(bad_reversions, shock_ops), "Signal Interpretation Quality", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Athey & Bagwell: Rationality, Deception, Cooperation, Reasoning."""
        if not game_results: return {}
        game_info = game_results[0]
        
        low_ops, low_truth = 0, 0
        high_ops, high_lies = 0, 0
        strat_ops, total_truth = 0, 0
        reason_ops, reason_success = 0, 0

        for r in game_results:
            rounds = r.game_data.get('rounds', [])
            for i, rd in enumerate(rounds):
                if rd.get('period', 1) % 2 != 0: # Odd/Strategic
                    strat_ops += 1
                    true_c = rd.get('player_true_costs', {}).get(player_id)
                    rep = rd.get('actions', {}).get(player_id, {}).get('report')
                    
                    if not true_c or not rep: continue
                    
                    # 3. Cooperation (Truthfulness)
                    if rep == true_c: total_truth += 1
                    
                    # 1. Rationality (Low Cost)
                    if true_c == 'low':
                        low_ops += 1
                        if rep == 'low': low_truth += 1
                    
                    # 2. Deception (High Cost)
                    elif true_c == 'high':
                        high_ops += 1
                        is_lying = (rep == 'low')
                        if is_lying: high_lies += 1
                        
                        # 4. Reasoning (Strategic Adaptation Accuracy)
                        if i >= 2:
                            prev_reps = [a.get('report') for pid, a in rounds[i-2].get('actions', {}).items() if pid != player_id]
                            if prev_reps:
                                reason_ops += 1
                                rivals_vuln = all(r == 'high' for r in prev_reps)
                                if (rivals_vuln and is_lying) or (not rivals_vuln and not is_lying):
                                    reason_success += 1

        return {
            'rationality': create_metric_result('rationality', self.safe_divide(low_truth, low_ops), "Low-Cost Truthfulness", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'deception': create_metric_result('deception', self.safe_divide(high_lies, high_ops), "Strategic Misrepresentation", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'cooperation': create_metric_result('cooperation', self.safe_divide(total_truth, strat_ops), "Productive Efficiency (Truth)", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', self.safe_divide(reason_success, reason_ops), "Strategic Adaptation Accuracy", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }