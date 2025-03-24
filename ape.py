import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Configure Gemini API (Replace with your actual API key)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Step 1: Generate Candidate Instructions
def generate_instructions(task_description, num_candidates=5):
    """Generates multiple candidate instructions for a given task using Gemini."""
    prompt = f"""
    You are an expert prompt engineer. Your task is to generate {num_candidates} different, high-quality instructions 
    for the following task: {task_description}
    
    Provide each instruction as a separate numbered list.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text.split("\n") if response.text else []

# Step 2: Evaluate Instructions by Testing on a Sample Input
def evaluate_instruction(instruction, sample_input):
    """Evaluates an instruction by testing it on a sample input."""
    prompt = f"""
    Instruction: {instruction}
    Sample Input: {sample_input}
    
    Execute the instruction on the given input and return the response.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Step 3: Select the Best Instruction
def select_best_instruction(task_description, sample_input):
    """Generates multiple instructions, evaluates them, and selects the best one."""
    candidates = generate_instructions(task_description)
    
    if not candidates:
        return "Failed to generate instructions."

    evaluations = {}
    
    for instruction in candidates:
        result = evaluate_instruction(instruction, sample_input)
        evaluations[instruction] = result

    # Select the best instruction based on response quality (Here, we use length as a proxy for testing)
    best_instruction = max(evaluations, key=lambda instr: len(evaluations[instr]))
    
    return best_instruction, evaluations[best_instruction]

# Example Task
task_description = "Summarize a paragraph in one sentence."
sample_input = "The remarkable success of pretrained language models has motivated the study of what kinds of knowledge these models learn during pretraining. Reformulating tasks as fill-in-the-blanks problems (e.g., cloze tests) is a natural approach for gauging such knowledge, however, its usage is limited by the manual effort and guesswork required to write suitable prompts. To address this, we develop AutoPrompt, an automated method to create prompts for a diverse set of tasks, based on a gradient-guided search. Using AutoPrompt, we show that masked language models (MLMs) have an inherent capability to perform sentiment analysis and natural language inference without additional parameters or finetuning, sometimes achieving performance on par with recent state-of-the-art supervised models. We also show that our prompts elicit more accurate factual knowledge from MLMs than the manually created prompts on the LAMA benchmark, and that MLMs can be used as relation extractors more effectively than supervised relation extraction models. These results demonstrate that automatically generated prompts are a viable parameter-free alternative to existing probing methods, and as pretrained LMs become more sophisticated and capable, potentially a replacement for finetuning."

# Run APE
best_prompt, output = select_best_instruction(task_description, sample_input)
print("\nBest Generated Prompt:", best_prompt)
print("\nExample Output:", output)
