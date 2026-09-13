from aegisflow.data.generator_v1 import generate_dataset
from aegisflow.envs.aegisflow_env import AegisFlowEnv

def test_generator_shape():
    df = generate_dataset(1000)
    assert len(df) == 1000
    assert df["required_level"].between(0, 3).all()

def test_environment_step():
    df = generate_dataset(100)
    env = AegisFlowEnv(df, episode_length=4)
    state, _ = env.reset(seed=1)
    assert state.shape == (14,)
    next_state, reward, terminated, truncated, info = env.step(2)
    assert next_state.shape == (14,)
    assert isinstance(reward, float)
    assert "security_requirement" in info
