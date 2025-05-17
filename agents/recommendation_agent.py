import google.generativeai as genai
import os
import pickle

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class RecommendationAgent:
    def __init__(self, model_name="gemini-2.0-flash", memory_file="recommendation_memory.pkl"):
        self.model = genai.GenerativeModel(model_name)
        self.memory_file = memory_file
        self.memory = self.load_memory()
    
    def load_memory(self):
        try:
            with open(self.memory_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.memory, f)
    
    def recommend(self, user_preferences):
        """Generates personalized recommendations based on user preferences."""
        prompt = f"""
        User Preferences: {user_preferences}
        Past Interactions: {self.memory.get(user_preferences, "No past data")}
        
        Recommend three personalized items.
        """
        response = self.model.generate_content(prompt)
        recommendation = response.text.strip()
        self.memory[user_preferences] = recommendation
        self.save_memory()
        return recommendation

# Example Usage
agent = RecommendationAgent()
print(agent.recommend("Tech gadgets, AI books, productivity tools"))
