import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import sympy as sp
# from helpers import banzhaf, banzhaf_fast, distance, generalized_banzhaf, make_distribution, round_vector, make_distribution
import math

from helpers import make_distribution

class SimpleGame:
    """
        Given a population (target) and a quota, tries to find a simple game or WVG
        with the given quota that has powers as close as possible to the target.

        Takes as input: 
        - target: The populations to be approximated as closely as possible. Has to sum to 1.
        - quota: The quota for which we want to find a WVG. Ignored if wvg = False.
        - wvg: Whether we are looking for a weighted voting game or a simple game.
        """
    def __init__(self, quota, target, wvg = True) -> None:
        n =  len(target)
        self.n = n
        self.quota = quota
        # The largest sum of weights to consider in the LP.  
        # M = n*((n+1))**((n+1)/2)/(2**n) is sufficient to represent any WVG by Kawana-Matsui 2021, 'Trading Transforms of Non-weighted Simple Games and Integer Weights of Weighted Simple Games'
        # multiplied by 2 to account for the 'integrality gap' in our quota constraints.
        self.M = 2*n*((n+1))**((n+1)/2)/(2**n) 

        target.sort()
        self.target = target
        if all(type(t) == int for t in target):
            self.T = sum(target)
            self.int_target = True
        else: 
            # If we don't use an integer target, the find_game method only shows that the found weights are optimal up to some arbitrarily small eps, but not exactly optimal
            if not math.isclose(sum(target), 1): raise ValueError('Target not normalized.')
            self.int_target = False

        self.wvg = wvg
        self.current_alpha = None
        self.veto_constraints_added = False

    def make_ilp(self, alpha = 2, maximal_total_weight = None, propensity = None):
        """
        Make the ILP to confirm whether there exists a WVG with L1 distance at most alpha to the target,
        with given propensity. If no propensity is specified, 1/2 is used.
        """
        if maximal_total_weight == None: maximal_total_weight = self.M

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('LogToConsole', 0)
  
        env.start()
        model = gp.Model("weights_check", env=env)
    
        # Weight variables
        if self.wvg: 
            w = model.addVars(self.n, lb=0,name="w", vtype=GRB.CONTINUOUS)
        # Swing count variables
        s = model.addVars(self.n, lb=0, name="s", vtype=GRB.CONTINUOUS)
        # Deviation variables: delta_i >= |si - di * s| 
        delta = model.addVars(self.n, name="delta", vtype=GRB.CONTINUOUS)
        # Total swings variable
        s_total = model.addVar(lb=0, name="s_total", vtype=GRB.CONTINUOUS)
        
        # Store variable references for binary search (when finding optimal game)
        self.delta_vars = delta
        self.s_total_var = s_total

        coalitions = [frozenset(combo) for i in range(self.n + 1) for combo in combinations(range(self.n), i)]
        is_winning = {}
        is_pivotal = {i : {} for i in range(self.n)}
        for coal in coalitions:
            is_winning[coal] = model.addVar(name=f"x_{str(coal)}", vtype=GRB.BINARY)
            for player in range(self.n):
                if player in coal: continue
                is_pivotal[player][coal] = model.addVar(name=f"y_{player},{str(coal)}", vtype=GRB.BINARY) # maybe could change this to continuous?

        # Constraint: s_total = sum(si) (Kurz's equation 16)
        model.addConstr(s_total == s.sum(), name="total_swings")

        # Total devitation constraint
        # Store reference to total_deviation constraint for binary search
        self.total_deviation_constr = model.addConstr(delta.sum() <= alpha * s_total, name="total_deviation")
        self.current_alpha = alpha
        
        for i in range(self.n):
            # Constraints: εi = |si - di * s_total| (Kurz's equations 18-19)
            if self.int_target:
                model.addConstr(self.T * s[i] <= self.target[i] * s_total + self.T * delta[i], name=f"dev_constraint_{i}_upper")
                model.addConstr(self.T * s[i] >= self.target[i] * s_total - self.T * delta[i], name=f"dev_constraint_{i}_lower")
            else:
                model.addConstr(s[i] <= self.target[i] * s_total + delta[i], name=f"dev_constraint_{i}_upper")
                model.addConstr(s[i] >= self.target[i] * s_total - delta[i], name=f"dev_constraint_{i}_lower")

            # Swings per player
            if propensity == None: model.addConstr(s[i] == gp.quicksum(is_pivotal[i].values()), name=f"total_swings_{i}")
            else: # Swings weighted by propensity
                if type(propensity) not in [sp.core.numbers.Rational, sp.core.numbers.Half]: raise ValueError("propensity should be rational for increased precision")
                prop_yes = propensity.numerator
                prop_no = propensity.denominator - propensity.numerator
                model.addConstr(s[i] == gp.quicksum(is_pivot * ((prop_yes)**(len(coal))) * ((prop_no)**(self.n - 1 - len(coal))) for coal, is_pivot in is_pivotal[i].items()), name=f"total_swings_{i}")

            for coal in is_pivotal[i]:
                model.addConstr(is_pivotal[i][coal] == is_winning[(coal | {i})] - is_winning[coal], name=f"pivotal_{i}_{str(coal)}")

        # The empty coalition is loosing, the grand coalition is winning.
        model.addConstr(is_winning[frozenset(range(self.n))] == 1, name="grand_winning")
        model.addConstr(is_winning[frozenset()] == 0, name="empty_loosing")

        # The weight constraints
        if self.wvg:
            if type(self.quota) == sp.core.numbers.Rational:
                quota_denom = self.quota.denominator
                quota_numer = self.quota.numerator
                for coal in coalitions:
                    # If coalition sum < quota * total_weight + 1, then is_winning = 0
                    model.addConstr(quota_numer * w.sum() -  quota_denom * sum(w[i] for i in coal) + quota_denom <=  quota_denom * (1 - is_winning[coal]) * maximal_total_weight, name=f"weights_{str(coal)}_lower")
                    # If coalition sum > quota * total_weight, then is_winning = 1
                    model.addConstr( quota_denom * sum(w[i] for i in coal) - quota_numer * w.sum() <= quota_denom * is_winning[coal] * maximal_total_weight, name=f"weights_{str(coal)}_upper")
            else:
                for coal in coalitions:
                    # If coalition sum < quota* total_weight + 1, then is_winning = 0
                    model.addConstr(self.quota * w.sum() -  sum(w[i] for i in coal) + 1 <=  (1 - is_winning[coal]) * maximal_total_weight, name=f"weights_{str(coal)}_lower")
                    # If coalition sum > quota* total_weight, then is_winning = 1
                    model.addConstr( sum(w[i] for i in coal) - self.quota * w.sum() <= is_winning[coal] * maximal_total_weight, name=f"weights_{str(coal)}_upper")

            model.addConstr(maximal_total_weight >= gp.quicksum(w[i] for i in range(self.n)), name="w_total_constraint")

            # These constraints aren't necessary but empirically make it faster. Since the target is sorted we know the optimal weights will be as well
            for i in range(self.n-1):
                model.addConstr(w[i+1] >= w[i], name=f"w_order_{i}")
        
        # These constraints aren't necessary in the WVG case but empirically make it faster
        for coal in coalitions:
            # If a coalition is winning, it is still winning if another voter joins
            for i in coal:
                model.addConstr(is_winning[coal] >= is_winning[frozenset(j for j in coal if j != i)],name=f"closed_{coal}_{i}")

                # These constraints aren't necessary at all but empirically make it faster IF THE LP IS INFESAIBLE
                if i-1 not in coal:
                    alternate_coal = frozenset(j for j in coal if j != i)
                    if i-1 >= 0: alternate_coal = alternate_coal | frozenset([i-1])
                    model.addConstr(is_winning[alternate_coal] <= is_winning[coal], name=f"closed_{coal}_{i}")



        model.update()
        self.model = model

        # print("Constraints in model (name and content):")
        # for c in model.getConstrs():
        #     print(f"{c.ConstrName}: {model.getRow(c)} {c.Sense} {c.RHS}")
        
        # Store additional variable references
        if self.wvg:
            self.w_vars = w
        else:
            self.w_vars = None
        self.is_winning_vars = is_winning
        self.s_total_var = s_total
        
    def _update_alpha_constraint(self, alpha):
        """Update the total_deviation constraint with new alpha value."""
        # # Remove old constraint
        # self.model.remove(self.total_deviation_constr)
        # # Add new constraint with updated alpha
        # self.total_deviation_constr = self.model.addConstr(
        #     self.delta_vars.sum() <= alpha * self.s_total_var,
        #     name="total_deviation"
        # )

        # The constraint is: delta.sum() <= alpha * s_total
        # In standard form: delta.sum() - alpha * s_total <= 0
        # So the coefficient of s_total is -alpha
        # Use chgCoeff to update the coefficient efficiently
        self.model.chgCoeff(self.total_deviation_constr, self.s_total_var, -alpha)
        self.model.update()
        self.current_alpha = alpha

    def _add_no_veto_constraint(self):
        """Add constraints ensuring no player's veto player status changes."""
        for i in range(self.n):
            if self.int_target: is_veto_player = 1 if self.target[i] >= (1 - self.quota) * self.T  else 0
            else: is_veto_player = 1 if self.target[i] >= 1 - self.quota  else 0

            coalition_without_i = frozenset(j for j in range(self.n) if j != i)
            self.model.addConstr(
                self.is_winning_vars[coalition_without_i] == (1 - is_veto_player), 
                name=f"no_veto_{i}"
            )

            # These constraints aren't neccesary but may speed it up  
            if self.wvg: 
                if type(self.quota) == sp.core.numbers.Rational:
                    quota_denom = self.quota.denominator
                    quota_numer = self.quota.numerator
                    if is_veto_player:
                        self.model.addConstr(
                            quota_numer * self.w_vars.sum() >= quota_denom * sum(self.w_vars[j] for j in range(self.n) if j != i), 
                            name=f"no_veto_{i}_weight"
                        )
                    else:
                        self.model.addConstr(
                            quota_numer * self.w_vars.sum() + quota_denom <= quota_denom * sum(self.w_vars[j] for j in range(self.n) if j != i), 
                            name=f"no_veto_{i}_weight"
                        )
                else:
                    if is_veto_player:
                        self.model.addConstr(
                            self.quota * self.w_vars.sum() >= sum(self.w_vars[j] for j in range(self.n) if j != i), 
                            name=f"no_veto_{i}_weight"
                        )
                    else:
                        self.model.addConstr(
                            self.quota * self.w_vars.sum() + 1 <= sum(self.w_vars[j] for j in range(self.n) if j != i), 
                            name=f"no_veto_{i}_weight"
                        )
                

        self.model.update()
        self.veto_constraints_added = True

    def find_game(self, alpha = None, veto_constraint = False):
        """
        Check if the current LP is feasible and return a feasible assignment if it exists.
        Optimized for speed - only checks feasibility, not optimality.

        Args:
            alpha: The desired L1 accuracy
            veto_constraint: Whether to add veto constraints (default: False)
        
        Returns:
            dict with keys:
                - 'feasible': bool, whether the LP is feasible
                - 'assignment': dict or None
                    - If wvg=True: dict mapping player indices to weight values
                    - If wvg=False: dict mapping coalitions (frozensets) to is_winning values (0 or 1)
                - 'alpha': float, current alpha value used
                - 'veto_constraints_added': bool, whether veto constraints were added
        """
        if self.model is None:
            raise ValueError("LP model not created. Call make_ilp() first.")

        if alpha != None:
            self._update_alpha_constraint(alpha)

        # Add veto constraints if requested
        if veto_constraint and not self.veto_constraints_added:
            self._add_no_veto_constraint()
        
        # Set objective to 0 for feasibility-only check (faster)
        self.model.setObjective(0, GRB.MINIMIZE)
        
        # Optimize parameters for faster feasibility checking
        # Disable dual reductions for faster feasibility detection
        self.model.setParam('DualReductions', 0)
        # Keep presolve on as it can help with feasibility
        # Set method to automatic (let Gurobi choose fastest)
        self.model.setParam('Method', -1)
        
        # Optimize for feasibility (faster than full optimization)
        self.model.optimize()
        
        result = {
            'alpha': self.current_alpha,
            'veto_constraints_added': self.veto_constraints_added,
            'feasible': False,
            'assignment': None
        }
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            result['feasible'] = True
            
            if self.wvg:
                # Return weight assignment
                if self.w_vars is not None:
                    assignment = {i: self.w_vars[i].X for i in range(self.n)}
                    result['assignment'] = assignment
            else:
                # Return is_winning assignment
                assignment = {}
                for coal in self.is_winning_vars:
                    assignment[coal] = int(round(self.is_winning_vars[coal].X))
                result['assignment'] = assignment
        elif self.model.status == GRB.INFEASIBLE:
            result['feasible'] = False
        else:
            # Other statuses (unbounded, etc.)
            raise ValueError('Unexpected Error.')
            result['feasible'] = False
        
        return result

    def lp_lower_bound(self):
        """
        Relax all binary variables to continuous [0,1] and check feasibility.
        This provides a lower bound - if the relaxed LP is infeasible, the original ILP is also infeasible.
        
        Note: This lower bound is empirically not very tight nor really useful.

        Returns:
            dict with keys:
                - 'feasible': bool, whether the relaxed LP is feasible
                - 'assignment': dict or None
                    - If wvg=True: dict mapping player indices to weight values
                    - If wvg=False: dict mapping coalitions (frozensets) to is_winning values (in [0,1])
                - 'alpha': float, current alpha value used
                - 'veto_constraints_added': bool, whether veto constraints were added
        """
        if self.model is None:
            raise ValueError("LP model not created. Call make_ilp() first.")
        
        # Copy the model to avoid modifying the original
        relaxed_model = self.model.copy()
        
        # Build mapping of variable names to relaxed variables for easier access
        relaxed_vars_by_name = {var.VarName: var for var in relaxed_model.getVars()}
        
        # Relax all binary variables to continuous [0,1]
        for var in relaxed_model.getVars():
            if var.VType == GRB.BINARY:
                var.setAttr('VType', GRB.CONTINUOUS)
                var.setAttr('LB', 0.0)
                var.setAttr('UB', 1.0)
        
        relaxed_model.update()
        
        # Set objective to 0 for feasibility-only check (faster)
        relaxed_model.setObjective(0, GRB.MINIMIZE)
        
        # Optimize parameters for faster feasibility checking
        relaxed_model.setParam('DualReductions', 0)
        relaxed_model.setParam('Method', -1)
        
        # Optimize for feasibility
        relaxed_model.optimize()
        
        result = {
            'alpha': self.current_alpha,
            'veto_constraints_added': self.veto_constraints_added,
            'feasible': False,
            'assignment': None
        }
        
        if relaxed_model.status == GRB.OPTIMAL or relaxed_model.status == GRB.SUBOPTIMAL:
            result['feasible'] = True
            
            if self.wvg:
                # Return weight assignment
                if self.w_vars is not None:
                    assignment = {}
                    for i in range(self.n):
                        # Weight variables are named "w[0]", "w[1]", etc.
                        w_var = relaxed_vars_by_name.get(f"w[{i}]")
                        if w_var is not None:
                            assignment[i] = w_var.X
                    result['assignment'] = assignment
            else:
                # Return is_winning assignment (now continuous values in [0,1])
                assignment = {}
                for coal in self.is_winning_vars:
                    # is_winning variables are named "x_{str(coal)}"
                    x_var = relaxed_vars_by_name.get(f"x_{str(coal)}")
                    if x_var is not None:
                        assignment[coal] = x_var.X
                result['assignment'] = assignment
        elif relaxed_model.status == GRB.INFEASIBLE:
            result['feasible'] = False
        else:
            # Other statuses (unbounded, etc.)
            raise ValueError(f"Unexpected model status: {relaxed_model.status}")
        
        # Clean up the relaxed model
        relaxed_model.dispose()
        
        return result

    def find_optimal_game(self, accuracy=None, veto_constraint=False):
        """
        Perform binary search to find the smallest alpha (up to specified accuracy) 
        for which the ILP is satisfiable.
        
        Args:
            accuracy: The desired accuracy for the binary search
            veto_constraint: Whether to add veto constraints (default: False)
        
        Returns:
            dict with keys:
                - 'alpha': float, the optimal alpha found
                - 'assignment': dict, the assignment from find_game() at optimal alpha
                - 'feasible': bool, whether a feasible solution was found
        """
        if self.model is None:
            raise ValueError("LP model not created. Call make_ilp() first.")

        if accuracy == None: accuracy = 1/(self.n * 2**(self.n))**2
        
        # Add veto constraints if requested
        if veto_constraint and not self.veto_constraints_added:
            self._add_no_veto_constraint()
        
        # First, check if alpha=0 is feasible
        self._update_alpha_constraint(0.0)
        result_zero = self.find_game()
        
        if result_zero['feasible']:
            return {
                'alpha': 0.0,
                'assignment': result_zero['assignment'],
                'feasible': True
            }
        
        # Binary search bounds
        # Start with alpha=0 (infeasible) and find an upper bound
        alpha_low = 0.0
        alpha_high = 2 # alpha = 2 should always be feasible

        self._update_alpha_constraint(alpha_high)
        result = self.find_game()
        if not result['feasible']:
            raise ValueError("alpha=2 should always be feasible. Investiage why not.")            
        
        # # Try increasing alpha values to find an upper bound
        # test_alpha = 0.1
        # max_test_alpha = 10.0
        # while test_alpha <= max_test_alpha:
        #     self._update_alpha_constraint(test_alpha)
        #     result = self.find_game()
        #     if result['feasible']:
        #         alpha_high = test_alpha
        #         break
        #     test_alpha *= 2
        
        # # If no feasible alpha found in reasonable range, return infeasible
        # if alpha_high is None:
        #     return {
        #         'alpha': None,
        #         'assignment': None,
        #         'feasible': False
        #     }
        
        # Binary search to find the smallest feasible alpha
        best_alpha = alpha_high
        best_assignment = result['assignment']
        
        while alpha_high - alpha_low > accuracy:
            alpha_mid = (alpha_low + alpha_high) / 2.0
            self._update_alpha_constraint(alpha_mid)
            result = self.find_game()
            
            if result['feasible']:
                # Found feasible solution, try smaller alpha
                alpha_high = alpha_mid
                best_alpha = alpha_mid
                best_assignment = result['assignment']
            else:
                # Infeasible, need larger alpha
                alpha_low = alpha_mid
        
        return {
            'alpha': best_alpha,
            'assignment': best_assignment,
            'feasible': True
        }

  
if __name__ == "__main__":
    # Example where the optimum induces an undeserving veto player.
    quota = sp.Rational(3,4)
    game = SimpleGame(quota, [1,1,1,1,1,1,1,2], True)
    game.make_ilp(propensity = sp.Rational(11,20))
    print(game.find_optimal_game(accuracy = 0.0001, veto_constraint = False))
    print(game.find_optimal_game(accuracy = 0.0001, veto_constraint = True))


    ontario_population = [1641,1668,2284,2403,2644,2740,3360,3473,3640,3679,3921,3931,4106,5140,5210,5436,6637,9404,11109,14170,15860]
    quota = sp.Rational(3,4)
    game = SimpleGame(quota, ontario_population, True)
    game.make_ilp() 
    # This is the command we want to run
    # We know there exists a solution with a veto player with alpha = 0.00044..., 
    # so we want to verify no solution without inducing a veto player can get
    # better L1 norm:

    # print(game.find_game(alpha = 0.00045)) 