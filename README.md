# Plexe

Create ML models from natural language descriptions.

## Installation

```bash
pip install plexe
```

## Usage

First, set your API key:
```bash
export PLEXE_API_KEY=your_api_key_here
```

Then use it in your code:

```python
import plexe

# Create a model
model_id, version, desc = plexe.create(
    "Create a model that predicts sentiment from text"
)

# Make predictions
result = plexe.run(
    model_id=model_id,
    text_input="This product is amazing!"
)
print(result)

# Batch predictions
results = plexe.batch_run(
    model_id=model_id,
    inputs=[
        {"text": "This is great"},
        {"text": "This is terrible"}
    ]
)

# Async support
async def main():
    model_id, version, desc = await plexe.acreate(
        "Create a classifier for text"
    )
    result = await plexe.arun(model_id, text_input="Test")
```
