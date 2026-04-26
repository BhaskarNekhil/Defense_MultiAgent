"""
Smoke tests for Defense-AI environment.
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DefenseAction, DefenseObservation, DefenseState
from defense_env.environment import DefenseEnvironment
from defense_env.tasks import grade_easy, grade_medium, grade_hard


def test_reset_returns_observation():
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env = DefenseEnvironment(task_id=task_id)
        obs = env.reset()
        assert isinstance(obs, DefenseObservation)
        assert obs.task_id == task_id
        assert len(obs.radar_contacts) > 0
        assert obs.done is False
        assert obs.steps_taken == 0


def test_state_returns_state():
    env = DefenseEnvironment(task_id="task_easy")
    env.reset()
    state = env.state
    assert isinstance(state, DefenseState)
    assert state.step_count == 0


def test_step_scan_returns_reward():
    env = DefenseEnvironment(task_id="task_easy")
    obs = env.reset()
    tid = obs.threats_in_scope[0]
    action = DefenseAction(action_type="scan_airspace", target_id=tid)
    obs2 = env.step(action)
    assert isinstance(obs2, DefenseObservation)
    assert obs2.steps_taken == 1
    assert obs2.reward != 0.0


def test_submit_report_ends_episode():
    env = DefenseEnvironment(task_id="task_easy")
    obs = env.reset()
    action = DefenseAction(
        action_type="submit_report",
        engagements=[],
        decisions_summary="test"
    )
    obs2 = env.step(action)
    assert obs2.done is True


def test_reward_in_valid_range():
    env = DefenseEnvironment(task_id="task_medium")
    obs = env.reset()
    for tid in obs.threats_in_scope:
        action = DefenseAction(action_type="scan_airspace", target_id=tid)
        obs = env.step(action)
        assert -0.99 <= obs.reward <= 0.99


def test_graders_return_valid_scores():
    # grade_easy
    score = grade_easy(
        [{"target_id": "e1", "action": "engage_machine_gun"}],
        "radar scan iff enemy engage machine gun",
        ["e1"], [], []
    )
    assert 0.01 <= score <= 0.99

    # grade_medium
    score = grade_medium(
        [{"target_id": "e1", "action": "engage_machine_gun"},
         {"target_id": "m1", "action": "launch_abm"}],
        "radar scan iff missile intercept abm enemy machine gun friendly",
        ["e1"], [], ["m1"]
    )
    assert 0.01 <= score <= 0.99

    # grade_hard
    score = grade_hard(
        [{"target_id": "e1", "action": "engage_machine_gun"},
         {"target_id": "m1", "action": "launch_abm"}],
        "radar iff spoofing discrimination missile abm machine gun",
        ["e1"], [], ["m1"], ["e1"]
    )
    assert 0.01 <= score <= 0.99


def test_no_fratricide_on_hold_fire():
    """hold_fire on friendly must NOT be counted as fratricide in grader."""
    score = grade_medium(
        [{"target_id": "f1", "action": "hold_fire"},
         {"target_id": "e1", "action": "engage_machine_gun"},
         {"target_id": "m1", "action": "launch_abm"}],
        "radar iff scan missile intercept abm machine gun enemy friendly",
        ["e1"], ["f1"], ["m1"]
    )
    # Should be high — perfect execution
    assert score >= 0.80, f"Expected >=0.80 but got {score}"
