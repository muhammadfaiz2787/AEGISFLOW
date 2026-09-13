import numpy as np
import torch

from aegisflow.config import TrainConfig
from aegisflow.rl.ddqn import DoubleDQNAgent


def main():

    print("=" * 70)
    print("AEGISFLOW DOUBLE DQN TEST")
    print("=" * 70)

    config = TrainConfig()

    agent = DoubleDQNAgent(config)

    print("\n[1] DEVICE")

    print("Device:", agent.device)

    print("\n[2] NETWORK")

    print(
        "Online parameters:",
        sum(
            p.numel()
            for p in agent.online_net.parameters()
        ),
    )

    print(
        "Target parameters:",
        sum(
            p.numel()
            for p in agent.target_net.parameters()
        ),
    )

    print("\n[3] INITIAL EPSILON")

    print(
        "Epsilon:",
        agent.epsilon,
    )

    print("\n[4] ACTION TEST")

    state = np.random.rand(
        config.state_dim
    ).astype(np.float32)

    for i in range(10):

        action = agent.select_action(
            state,
            training=True,
        )

        assert 0 <= action < config.action_dim

        print(
            f"Action {i + 1}: {action}"
        )

    print("\n[5] REPLAY TEST")

    for _ in range(
        config.min_replay
    ):

        state = np.random.rand(
            config.state_dim
        ).astype(np.float32)

        next_state = np.random.rand(
            config.state_dim
        ).astype(np.float32
        )

        action = np.random.randint(
            config.action_dim
        )

        reward = np.random.randn()

        done = np.random.rand() < 0.1

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

    print(
        "Replay size:",
        len(agent.replay),
    )

    print("\n[6] TRAINING STEP")

    loss = agent.train_step()

    print(
        "Loss:",
        loss,
    )

    assert loss is not None
    assert np.isfinite(loss)

    print("\n[7] TARGET UPDATE")

    agent.update_target()

    print(
        "Target network updated."
    )

    print("\n[8] EPSILON DECAY")

    old_epsilon = agent.epsilon

    agent.decay_epsilon()

    print(
        "Before:",
        old_epsilon,
    )

    print(
        "After:",
        agent.epsilon,
    )

    assert agent.epsilon < old_epsilon

    print("\n" + "=" * 70)
    print("DOUBLE DQN TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()