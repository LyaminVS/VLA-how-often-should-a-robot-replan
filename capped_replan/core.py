"""Reuse frozen execution while preserving explicit capped identities and reference gates."""

from adaptive_replan.core import run_episode as frozen_run_episode


class SchedulingView:
    """Only the scheduling dispatch changes; saved identity remains the real method."""

    def __init__(self, episode):
        self.episode = episode

    @property
    def method(self):
        return "dvac" if self.episode.adaptive else self.episode.method

    def __getattr__(self, name):
        return getattr(self.episode, name)

    def asdict(self):
        return self.episode.asdict()


class ReferenceEnvironment:
    def __init__(self, environment, reference):
        self.environment, self.reference = environment, reference

    def reset(self, episode):
        observation, info = self.environment.reset(episode)
        if episode.pair_id != self.reference["pair_id"]:
            raise ValueError("Reference pair identity differs from the requested episode")
        for key in ("initial_state_sha256", "initial_observation_sha256"):
            if info.get(key) != self.reference[key]:
                raise ValueError(f"Historical reference differs at reset: {key}")
        return observation, info

    def step(self, action):
        return self.environment.step(action)


class ReferencePolicy:
    def __init__(self, policy, reference):
        self.policy, self.reference = policy, reference
        self.first = True

    def reset(self):
        self.first = True
        self.policy.reset()

    @property
    def last_diagnostics(self):
        return self.policy.last_diagnostics

    def predict(self, observation, seed):
        if self.first and seed != self.reference["sampling_seed"]:
            raise ValueError("Historical reference first-call sampling seed differs")
        actions = self.policy.predict(observation, seed)
        if self.first:
            if actions != self.reference["actions"]:
                raise ValueError("Historical reference first complete action chunk differs; no action executed")
            self.first = False
        return actions


def run_episode(env, policy, episode, *, selector=None, reference=None, record=None, **kwargs):
    if episode.adaptive:
        if selector is None or selector.execution_cap != episode.execution_cap:
            raise ValueError("Adaptive episode requires a selector with the declared execution cap")
    if reference is not None:
        env = ReferenceEnvironment(env, reference)
        policy = ReferencePolicy(policy, reference)

    def annotate(row):
        decision = selector.last_decision if episode.adaptive else None
        row["raw_chosen_k"] = decision["raw_chosen_k"] if decision else episode.k
        row["execution_cap"] = episode.execution_cap
        row["cap_applied"] = decision["cap_applied"] if decision else False
        return row

    def recording(frame, row, outcome):
        record(frame, annotate(dict(row)), outcome)

    summary, rows, chunks = frozen_run_episode(
        env, policy, SchedulingView(episode), selector=selector,
        record=recording if record is not None else None, **kwargs,
    )
    for chunk in chunks:
        decision = chunk["decision"]
        if not episode.adaptive:
            decision.update(raw_chosen_k=episode.k, execution_cap=None, cap_applied=False)
        for row in rows[chunk["step"]:chunk["step"] + chunk["executed_steps"]]:
            row.update({key: decision[key] for key in ("raw_chosen_k", "execution_cap", "cap_applied")})
    summary["mean_raw_chosen_k"] = sum(chunk["decision"]["raw_chosen_k"] for chunk in chunks) / len(chunks)
    summary["cap_applied_calls"] = sum(chunk["decision"]["cap_applied"] for chunk in chunks)
    if reference is not None:
        summary.update(reference_pair_id=reference["pair_id"], reference_initial_state_match=True,
                       reference_first_prediction_equal=True, reference_max_absolute_action_error=0.0)
    return summary, rows, chunks
