```bash
git clone https://github.com/hieubnt235/petorch.git
cd petorch
# Install pytorch for your cuda version ...

uv sync
pytest -vv

```
## Quickstart
1. AdaptedLayer and BaseAdapter subclass. See example in [here](https://github.com/hieubnt235/petorch/blob/master/src/petorch/prebuilt/lora/linear.py) .
2. Define the config, which is define "where to apply adapter in model" and "what adapter to apply". See example [here](https://github.com/hieubnt235/petorch/blob/master/src/petorch/prebuilt/configs.py).
3. Create model and use `AdapterApi`'s static methods to manipulate your model. See [here](https://github.com/hieubnt235/petorch/blob/master/tests/adapter/test_base_and_api.py) for more detail how to use it and what's their tolorates.
