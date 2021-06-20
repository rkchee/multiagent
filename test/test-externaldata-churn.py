train \
    --run=DQN \
    --env=CartPole-v0 \
    --config='{
        "input": "/tmp/demo-out",
        "input_evaluation": [],
        "explore": false}'