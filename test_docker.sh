docker run --rm \
-v "$(pwd)"/data/ddos:/data/ddos \
-v "$(pwd)"/data/inputs:/data/inputs \
-v "$(pwd)"/data/outputs:/data/outputs \
-v "$(pwd)"/eda.py:/app/eda.py \
-e DIDS='["8f67E08be5dD941a701c2491E814535522c33bC2"]' \
rogargon/pandas-profiling python3 eda.py