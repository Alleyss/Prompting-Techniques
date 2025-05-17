import json
import google.generativeai as genai
class ECommerceAgent:
    def __init__(self, model_name="gemini-2.0-flash", memory_file="orders.json"):
        self.model = genai.GenerativeModel(model_name)
        self.memory_file = memory_file
        self.load_memory()
    
    def load_memory(self):
        try:
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = {}
    
    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f)
    
    def handle_order(self, customer_request):
        """Processes orders and provides shopping recommendations."""
        prompt = f"Customer Request: {customer_request}\nProvide relevant product recommendations."
        response = self.model.generate_content(prompt)
        order_info = response.text.strip()
        self.memory[customer_request] = order_info
        self.save_memory()
        return order_info

# Example Usage
ecom_agent = ECommerceAgent()
print(ecom_agent.handle_order("I need a new laptop for gaming."))
