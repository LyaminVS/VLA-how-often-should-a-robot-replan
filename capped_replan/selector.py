"""Apply an execution cap after the complete original DVAC decision."""

from adaptive_replan.selector import DVACSelector


class CappedDVACSelector:
    def __init__(self, execution_cap, *, horizon=50, tail_length=5, window_size=5, alpha=2.0, min_steps=1):
        if type(execution_cap) is not int or not min_steps <= execution_cap <= horizon:
            raise ValueError("execution_cap must be an integer in [min_steps, horizon]")
        self.execution_cap = execution_cap
        self.selector = DVACSelector(horizon=horizon, tail_length=tail_length,
                                     window_size=window_size, alpha=alpha, min_steps=min_steps)
        self.last_decision = None

    def reset(self):
        self.selector.reset()
        self.last_decision = None

    def select(self, variance):
        decision = self.selector.select(variance)
        raw = decision["chosen_k"]
        self.last_decision = {
            **decision, "raw_chosen_k": raw, "execution_cap": self.execution_cap,
            "cap_applied": raw > self.execution_cap, "chosen_k": min(raw, self.execution_cap),
        }
        return dict(self.last_decision)
