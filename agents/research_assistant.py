import sqlite3
import google.generativeai as genai
class ResearchAgent:
    def __init__(self, model_name="gemini-2.0-flash", db_file="research_memory.db"):
        self.model = genai.GenerativeModel(model_name)
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS research (query TEXT, response TEXT)")
    
    def search(self, topic):
        """Generates research-based summaries."""
        self.cursor.execute("SELECT response FROM research WHERE query=?", (topic,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        prompt = f"Conduct an in-depth research summary on: {topic}"
        response = self.model.generate_content(prompt)
        answer = response.text.strip()
        
        self.cursor.execute("INSERT INTO research (query, response) VALUES (?, ?)", (topic, answer))
        self.conn.commit()
        return answer

# Example Usage
research_agent = ResearchAgent()
print(research_agent.search("Impact of AI in healthcare"))
