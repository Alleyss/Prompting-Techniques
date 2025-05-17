import pandas as pd
import google.generativeai as genai
class ReportingAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.data = pd.DataFrame(columns=["Query", "Report"])

    def generate_report(self, query):
        """Analyzes data and generates a report."""
        prompt = f"Analyze data for: {query}\nGenerate a professional report."
        response = self.model.generate_content(prompt)
        report = response.text.strip()
        self.data = self.data.append({"Query": query, "Report": report}, ignore_index=True)
        return report

# Example Usage
report_agent = ReportingAgent()
print(report_agent.generate_report("Market trends in renewable energy"))
