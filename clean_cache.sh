find . -type d -name "__pycache__" -exec rm -r {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
rm ~/.cache/huggingface/datasets/* -rf
