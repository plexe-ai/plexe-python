<div align="center">

# PlexeAI


<img src="docs/img/plexe-logo.svg" alt="PlexeAI Logo" width="100" height="100"/>

### Create ML models from natural language descriptions

[![PyPI version](https://badge.fury.io/py/plexe.svg)](https://badge.fury.io/py/plexe)
[![Python Versions](https://img.shields.io/pypi/pyversions/plexe.svg)](https://pypi.org/project/plexe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🚀 Features

- 🤖 **AI-Powered Model Creation** - Build ML models using natural language descriptions
- 📊 **Automated Training** - Upload your data and let PlexeAI handle the rest
- ⚡ **Async Support** - Built-in async interfaces for high-performance applications
- 🔄 **Batch Processing** - Efficient batch prediction capabilities
- 🛠️ **Simple API** - Intuitive interface for both beginners and experts

## 📦 Installation

```bash
pip install plexe
```

## 🏃‍♂️ Quickstart

```python
import plexe

# Create a model in seconds
model_version = plexe.build(
    goal="predict customer churn based on usage patterns",
    model_name="churn-predictor",
    data_files="customer_data.csv"
)

# Make predictions
result = plexe.infer(
    model_name="churn-predictor",
    model_version=model_version,
    input_data={
        "usage": 100,
        "tenure": 12,
        "plan_type": "premium"
    }
)
```

## 🎯 Example Use Cases

- 📈 **Churn Prediction**: Predict customer churn using historical data
- 🏷️ **Classification**: Categorize text, images, or any structured data
- 📊 **Regression**: Predict numerical values like sales or pricing
- 🔄 **Time Series**: Forecast trends and patterns in sequential data

## 🔥 Advanced Usage

### Batch Predictions

```python
results = plexe.batch_infer(
    model_name="churn-predictor",
    model_version=model_version,
    inputs=[
        {"usage": 100, "tenure": 12, "plan_type": "premium"},
        {"usage": 50, "tenure": 6, "plan_type": "basic"}
    ]
)
```

### Async Support

```python
async def main():
    model_version = await plexe.abuild(
        goal="predict customer churn",
        model_name="churn-predictor",
        data_files="customer_data.csv"
    )
    
    result = await plexe.ainfer(
        model_name="churn-predictor",
        model_version=model_version,
        input_data={"usage": 100, "tenure": 12}
    )
```

### Direct Client Usage

```python
from plexe import PlexeAI

with PlexeAI(api_key="your_api_key_here") as client:
    # Upload data
    upload_id = client.upload_files("customer_data.csv")
    
    # Create and use model
    model_version = client.build(
        goal="predict customer churn",
        model_name="churn-predictor",
        upload_id=upload_id
    )
```

## 📚 Documentation

Check out our [comprehensive documentation](https://docs.plexe.ai) for:
- Detailed API reference
- Advanced usage examples
- Best practices
- Tutorials and guides

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/plexe-ai/plexe
cd plexe

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Plexe AI](https://plexe.ai)

</div>
