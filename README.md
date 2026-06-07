# Wine Quality Algorithmic Prediction System

An end-to-end modular MLOps production pipeline designed to ingest, validate, transform, and evaluate chemical wine properties to predict quality metrics. Built entirely with a modular Python package structure, this system utilizes ElasticNet regression modeling, integrates automated logging/telemetry, and features an interactive Flask user interface.

## 🚀 Key Architectural Features
* **Modular Pipeline Design**: Separated cleanly into independent stages (`Data Ingestion`, `Validation`, `Transformation`, `Training`, and `Evaluation`).
* **Strict Schema Verification**: Automatically verifies production data integrity against explicit types and strict column properties prior to training runs.
* **Production-Grade Web Layer**: Served using a decoupled configuration wrapper running a stable WSGI web server.

---

## 🛠️ Operational Workflow Blueprint
The system builds components sequentially via the following execution steps:
1. **`config/config.yaml`** - Handles centralized root folder asset allocation paths.
2. **`schema.yaml`** - Controls target variables and feature matrix properties.
3. **`params.yaml`** - Stores non-hardcoded ElasticNet hyperparameters ($\alpha$, $l1\_ratio$).
4. **Configuration / Entity Layers** - Serializes configuration blocks into immutable `dataclass` formats using `ConfigBox`.
5. **Components Pipeline** - Independently executes data gathering, verification checks, data splits, and model serialization.

---

## 💻 Local Setup & Execution Guide

### Step 1: Initialize Your Environment
Ensure you have Anaconda or Miniconda running in your terminal environment, then run:
```bash
# Create a dedicated isolated workspace environment
conda create -n mlproj python=3.8 -y
conda activate mlproj


# Install core dependencies along with local package binding constraints
pip install -r requirements.txt


# Boot the web application locally
python app.py