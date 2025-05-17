import google.generativeai as genai
import os
import argparse
# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class ReflexionAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)  # Use the correct model API
        self.memory = []  # Store past interactions for context
    
    def actor(self, observation):
        """Generates actions based on observations using Chain-of-Thought (CoT) and ReAct."""
        prompt = f"""
        Observation: {observation}
        Memory: {self.memory}
        
        Based on the observation and past experiences, determine the best action.
        """
        response = self.model.generate_content(prompt)
        return response.text.strip() if response.text else "No response generated"
    
    def evaluator(self, trajectory):
        """Evaluates the generated trajectory and returns a reward score."""
        prompt = f"""
        Given the following trajectory:
        {trajectory}
        
        Evaluate the effectiveness of the response on a scale of 1-10, with 10 being the best.
        """
        response = self.model.generate_content(prompt)
        try:
            return int(response.text.strip()) if response.text.isdigit() else 5
        except ValueError:
            return 5  # Default score if parsing fails
    
    def self_reflection(self, trajectory, score):
        """Generates feedback for self-improvement."""
        prompt = f"""
        Given the trajectory:
        {trajectory}
        
        And the score: {score}
        
        Provide constructive feedback on how to improve future actions.
        """
        response = self.model.generate_content(prompt)
        feedback = response.text.strip() if response.text else "No feedback generated"
        self.memory.append(feedback)  # Store feedback in memory
        return feedback
    
    def run(self, observation):
        """Runs the full reflexion cycle."""
        action = self.actor(observation)
        print(f"Action: {action}")
        
        score = self.evaluator(action)
        print(f"Score: {score}")
        
        feedback = self.self_reflection(action, score)
        print(f"Feedback: {feedback}")
        
        return action, score, feedback

# Example usage
if __name__ == "__main__":
    agent = ReflexionAgent()
    parser = argparse.ArgumentParser(description='AI Code Debugger')
    parser.add_argument('arg_str', help='error or code snippet')
    args = parser.parse_args()
    arg_str = args.arg_str
    observation = "The user needs help with debugging a Python script."+arg_str
    agent.run(observation)
