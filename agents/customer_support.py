import google.generativeai as genai
import os
import sqlite3
import json

# -------------------------------
# Setup Gemini API
# -------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("Please set your GOOGLE_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Long-Term Memory: SQLite for Conversation History & Training Data
# -------------------------------
class LongTermMemory:
    def __init__(self, db_file="agent_memory.db"):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.setup_db()

    def setup_db(self):
        """Create necessary tables if they do not exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                user_id TEXT,
                query TEXT,
                response TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                info_type TEXT PRIMARY KEY, 
                content TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS offers (
                offer_id TEXT PRIMARY KEY,
                description TEXT,
                discount_percentage REAL
            )
        """)
        self.conn.commit()

    def store_conversation(self, user_id, query, response):
        """Store user queries and AI responses."""
        self.cursor.execute("INSERT INTO conversation (user_id, query, response) VALUES (?, ?, ?)",
                            (user_id, query, response))
        self.conn.commit()

    def get_recent_conversations(self, user_id, limit=3):
        """Retrieve recent conversation history for a user."""
        self.cursor.execute("SELECT query, response FROM conversation WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                            (user_id, limit))
        return self.cursor.fetchall()

    def add_or_update_training_data(self, info_type, content):
        """Insert new information or update existing info dynamically."""
        self.cursor.execute("""
            INSERT INTO training_data (info_type, content) 
            VALUES (?, ?) 
            ON CONFLICT(info_type) DO UPDATE SET content = excluded.content
        """, (info_type, content))
        self.conn.commit()

    def get_training_data(self, info_type):
        """Retrieve updated training data from the database."""
        self.cursor.execute("SELECT content FROM training_data WHERE info_type=?", (info_type,))
        result = self.cursor.fetchone()
        return result[0] if result else "No data available."

    def add_or_update_offer(self, offer_id, description, discount_percentage):
        """Store or update available offers."""
        self.cursor.execute("""
            INSERT INTO offers (offer_id, description, discount_percentage)
            VALUES (?, ?, ?)
            ON CONFLICT(offer_id) DO UPDATE SET description = excluded.description, discount_percentage = excluded.discount_percentage
        """, (offer_id, description, discount_percentage))
        self.conn.commit()

    def get_offer_discount(self, offer_id):
        """Retrieve discount percentage for a specific offer."""
        self.cursor.execute("SELECT discount_percentage FROM offers WHERE offer_id=?", (offer_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

# -------------------------------
# Customer Support Agent
# -------------------------------
class CustomerSupportAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.memory = LongTermMemory()

    def plan_response(self, user_query, user_id):
        """
        Fetch dynamically updated training data and use it in the response generation.
        """
        recent_conv = self.memory.get_recent_conversations(user_id)
        history_text = "\n".join([f"User: {q}\nAgent: {r}" for q, r in recent_conv]) if recent_conv else "None"

        product_info = self.memory.get_training_data("product")
        offers_info = self.memory.get_training_data("offer")
        refund_info = self.memory.get_training_data("refund")

        prompt = f"""
You are a customer support AI agent for ElectroShop, handling orders, products, refunds, and discounts.
You have access to the following dynamically updated information:

- Product Information: {product_info}
- Current Offers: {offers_info}
- Refund Policy: {refund_info}

Recent Conversation History:
{history_text}

User Query: {user_query}

Follow these steps:
1. Identify if the query is about a product, offer, refund, discount, or general support.
2. Retrieve relevant information from the latest database updates.
3. If the user claims to have an eligible discount, apply it and calculate the new price.
4. Generate a well-structured response with a professional tone.

Provide a concise plan in green followed by the final answer in yellow.
        """
        response = self.model.generate_content(prompt)
        full_response = response.text.strip()

        if "\n" in full_response:
            plan, final_answer = full_response.split("\n", 1)
        else:
            plan, final_answer = "Plan not provided", full_response

        return f"\033[32m{plan}\033[0m", f"\033[33m{final_answer}\033[0m"

    def update_training(self, info_type, new_content):
        self.memory.add_or_update_training_data(info_type, new_content)

    def update_offer(self, offer_id, description, discount_percentage):
        self.memory.add_or_update_offer(offer_id, description, discount_percentage)

    def respond(self, user_id, user_query):
        plan, answer = self.plan_response(user_query, user_id)
        self.memory.store_conversation(user_id, user_query, answer)
        print(plan)
        print(answer)
        return answer

# -------------------------------
# Interactive CLI Prototype
# -------------------------------
def main():
    agent = CustomerSupportAgent()
    user_id = input("Enter your user ID: ")
    print("\nCustomer Support Chat Session. Type 'exit' to end.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Chat session ended.")
            break

        response = agent.respond(user_id, user_query)
        print(f"Agent: {response}")

        train_prompt = input("\nDo you want to update any information? (yes/no): ")
        if train_prompt.lower() == "yes":
            info_type = input("Enter info type (product/offer/refund): ")
            new_content = input("Enter the new info content: ")
            if info_type == "offer":
                discount_percentage = float(input("Enter discount percentage: "))
                agent.update_offer(info_type, new_content, discount_percentage)
            else:
                agent.update_training(info_type, new_content)
            print("Training data updated successfully.\n")

if __name__ == "__main__":
    main()
