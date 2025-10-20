import subprocess

subprocess.run(
    ["python", "-m", "src.algorithm"],
    cwd="/algorithm",
    check=True,
)
