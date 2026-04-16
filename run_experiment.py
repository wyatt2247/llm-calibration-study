# The comments are used to help me explain the code and make trouble shooting easier to solve issues that I encountered with the code.
# The libraries used in this code are:
#=================Imports & Libraries=================#
import os 
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


#=================Functions=====================#
#=================Configurations=====================#
API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ======== MODELS USED for this experiment ========

# ths are the models that are sent to OpenRouter via the API Call tells OpenRouter which models to call and use
MODELS = {
    "GPT-3.5-Turbo": "openai/gpt-3.5-turbo",
    "Llama-3.2-3B": "meta-llama/llama-3.2-3b-instruct",
    "Grok-4.1-Fast": "x-ai/grok-4.1-fast",
    "Mistral-small-3.1": "mistralai/mistral-small-3.1-24b-instruct",
    "Claude-3-haiku": "anthropic/claude-3-haiku",    
}

# ======== SUBJECTS USED ==========
SUBJECTS= [
    "high_school_world_history",
    "high_school_computer_science",
    "anatomy",
    "astronomy",
    "college_mathematics",
    "professional_law",
]

# ======== NUM of QUESTIONS and RUNS ========
NUM_QUESTIONS = 20
NUM_RUNS = 3
TEMPERATURE = 0.7 # adds variation between runs so consistency can be measured overall
MAX_TOKENS = 300

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

#================= Prompt ======================#

def build_prompt(question, choices):
    choices_text = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"""Question: {question}
{choices_text}

Respond with ONLY your answer letter (A,B,C, or D) on the first line.
On the second line, give your confidence score for the answer on a scale of 0 to 100.

Answer:"""


# ================== PARSING ======================#
def parse_response(response_text):
    if not response_text or not response_text.strip():
        return None, 50

    text = response_text.strip()

    answer = None
    confidence = 50 # safe fallback

    # Primary: Look for A/B/C/D followed by delimiter or space
    match = re.search(r'(?:^|\s|\n)([A-D])[)\s.:]', text, re.IGNORECASE)
    if match: 
        answer = match.group(1).upper()

    
    # Fallback: look for any lone A/B/C/D in the first 6 lines
    if not answer:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:6]:
            if line.upper().strip() in ['A','B','C','D']:
                answer = line.upper().strip()
                break
            match = re.search(r'\b([A-D])\b', line , re.IGNORECASE)
            if match: 
                answer = match.group(1).upper()
                break

    # === Confidence  ====
    conf_match = re.search(r'(?:confidence|conf|score)[:\s]*(\d+)', text, re.IGNORECASE)
    if conf_match:
        n = int(conf_match.group(1))
        if 0 <= n <= 100:
            confidence = n
    
    else: 
        # Any number between 0-100 as last resort
        numbers = re.findall(r'\b(\d{1,3}\b)', text)
        for num in numbers:
            n = int(num)
            if 0 <= n <= 100:
                confidence = n 
                break
    
    return answer, confidence


    # ================== LOAD QUESTIONS ======================#
def load_questions(subject):
    dataset = load_dataset("cais/mmlu", subject, split="test")
    samples = dataset.shuffle(seed=42).select(range(min(NUM_QUESTIONS, len(dataset))))
    questions = []
    for item in samples:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "correct_answer": chr(65 + item["answer"]),
        })
    return questions


# the script part that actually runs the experiment after getting the questions and models ready
# ================== RUN EXPERIMENT ======================#

def run_experiment():
    print("Starting experiment...")
    results = []
    total_calls= 0

    for subject in SUBJECTS:
        questions = load_questions(subject)
        print(f"\n=== {subject} ({len(questions)} questions) ===")

        for idx, q in enumerate(questions):
            print(f" Q{idx+1}/{len(questions)}")

            for model_name, model_id in MODELS.items():
                for run in range(NUM_RUNS):
                    try:
                        prompt = build_prompt(q["question"],q["choices"])
                        start = time.time()
                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[{"role":"user","content":prompt}],
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                        )
                        latency = time.time() - start

                        raw = response.choices[0].message.content

                        # Call parser
                        parsed = parse_response(raw)
                        
                        #Safety check to prevent crash
                        if parsed is None or len(parsed) != 2:
                            print(f"   Warning: Parser returned invalid result for {model_name} run {run+1}")
                            answer = None
                            confidence = 50
                        else: 
                            answer, confidence = parsed
                        is_correct = (answer == q["correct_answer"]) if answer else False

                        results.append({
                            "model": model_name,
                            "subject": subject,
                            "question": q["question"],
                            "correct_answer": q["correct_answer"],
                            "model_answer": answer,
                            "confidence": confidence,
                            "is_correct": is_correct,
                            "run": run + 1,
                            "latency": latency,
                            "raw_response": raw,
                            "timestamp": datetime.now().isoformat(),

                        })

                        total_calls += 1
                        time.sleep(2.0)
                    
                    except Exception as e:
                        print(f"   Error: {model_name} run {run+1} - {e}")
                        time.sleep(5)

    # outputs the results to the output file a csv file 

    df = pd.DataFrame(results)
    filename = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nDone! {total_calls} API calls saved to {filename}")
    return filename

if __name__ == "__main__":
    run_experiment()