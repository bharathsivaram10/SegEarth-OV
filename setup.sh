uv venv
uv pip install -r requirements.txt
uv pip install --no-build-isolation git+https://github.com/likyoo/SimFeatUp.git
uv run mim install "mmcv==2.1.0"