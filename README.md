# PlexeAI

Create ML models from natural language descriptions. Upload your data and describe your ML problem - PlexeAI handles the rest.

## Install

```bash
pip install plexeai
```

## Usage

```python
import plexeai

# Set via environment or pass to functions
# export PLEXE_API_KEY=your_api_key_here

# Build a model
experiment_id = plexeai.build(
    goal="predict customer churn based on usage patterns",
    data_files="customer_data.csv",  # Single file or list of files
    steps=3  # Number of improvement iterations
)

# Check build status
status = plexeai.get_status(experiment_id)
# {"status": "completed", "result": {...}}

# Make predictions
result = plexeai.infer(
    experiment_id=experiment_id,
    input_data={
        "usage": 100,
        "tenure": 12,
        "plan_type": "premium"
    }
)

# Batch predictions
results = plexeai.batch_infer(
    experiment_id=experiment_id,
    inputs=[
        {"usage": 100, "tenure": 12, "plan_type": "premium"},
        {"usage": 50, "tenure": 6, "plan_type": "basic"}
    ]
)

# Async support
async def main():
    experiment_id = await plexeai.abuild(
        goal="predict customer churn",
        data_files="customer_data.csv"
    )
    result = await plexeai.ainfer(
        experiment_id=experiment_id,
        input_data={"usage": 100, "tenure": 12}
    )
```

## Local Development

```bash
git clone https://github.com/plexe-ai/plexe
cd plexe
pip install -e ".[dev]"
pytest
```

## License

MIT License