# Databricks notebook source

# COMMAND ----------

import subprocess, sys, os, shutil, glob, tempfile

# Install deps
result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "torch", "numpy", "scipy", "polars", "pandas", "statsmodels", "pytest",
     "--quiet", "--no-warn-script-location"],
    capture_output=True, text=True
)
if result.returncode != 0:
    dbutils.notebook.exit("pip install failed: " + result.stderr[-1000:])

# Copy source files
workspace = "/Workspace/Users/pricing.frontier@gmail.com/insurance-drn"
install_path = tempfile.mkdtemp(prefix="drn_")

src_dst = os.path.join(install_path, "src", "insurance_drn")
tests_dst = os.path.join(install_path, "tests")
os.makedirs(src_dst, exist_ok=True)
os.makedirs(tests_dst, exist_ok=True)

for f in glob.glob(os.path.join(workspace, "src", "insurance_drn", "*.py")):
    shutil.copy2(f, src_dst)
for f in glob.glob(os.path.join(workspace, "tests", "*.py")):
    shutil.copy2(f, tests_dst)

# Run pytest
env = {**os.environ, "PYTHONPATH": os.path.join(install_path, "src")}
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True, text=True,
    cwd=install_path,
    env=env
)

full_output = result.stdout + "\n" + result.stderr

# Use notebook exit to surface the output
exit_msg = f"RETCODE={result.returncode}\n{full_output[-8000:]}"
if result.returncode != 0:
    dbutils.notebook.exit("FAIL: " + exit_msg)
else:
    dbutils.notebook.exit("PASS: " + exit_msg)
