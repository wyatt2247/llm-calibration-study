# LLM Calibration Study

This project evaluates the **confidence calibration and trustworthiness** of five commercial large language models using the MMLU benchmark.

### Models Tested
- Claude-3-Haiku
- GPT-3.5-Turbo
- Grok-4.1-Fast
- Llama-3.2-3B
- Mistral-small-3.1

### How to Run the Project

#### 1. Install Dependencies
```bash
pip install -r requirements.txt



2. Required Libraries

openai
pandas
numpy
matplotlib
seaborn
scikit-learn
datasets
python-dotenv

3. Running the Scripts
Step 1: Run the Experiment (This takes ~30 minutes to 2 hours)

python run_experiment.py

Step 2: Analyze and Visualize Results

python analyze.py
python visualize.py

Note: run_experiment.py makes 1,800 API calls to OpenRouter. Make sure you have sufficient credits and your API key set in .env.
Requirements

Python 3.9 or later
OpenRouter API key (set in .env file)

Project Structure

run_experiment.py – Runs the benchmark and saves results
analyze.py – Computes metrics (accuracy, calibration gap, ECE, Brier score, etc.)
visualize.py – Generates all plots and figures

