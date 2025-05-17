import google.generativeai as genai
class BookingAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.memory = {}

    def book(self, details):
        """Assists with travel and event bookings."""
        prompt = f"Booking Request: {details}\nProvide suitable options."
        response = self.model.generate_content(prompt)
        booking_info = response.text.strip()
        self.memory[details] = booking_info
        return booking_info

# Example Usage
booking_agent = BookingAgent()
print(booking_agent.book("Flight from New York to London on April 10"))
