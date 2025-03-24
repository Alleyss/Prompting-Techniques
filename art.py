import google.generativeai as genai
import sympy as sp
import requests
import os
from dotenv import load_dotenv
load_dotenv()
# Configure Gemini API (Replace with your actual API key)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Tool 1: Solve Equations using SymPy
def solve_equation(equation):
    """Solves algebraic equations using symbolic reasoning."""
    x = sp.Symbol('x')
    try:
        solution = sp.solve(equation, x)
        return f"Solution: {solution}"
    except Exception as e:
        return f"Error solving equation: {str(e)}"

# Tool 2: Get Weather Information (Free Weather API)
def get_weather(city):
    """Fetch real-time weather using a free API."""
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    return f"Weather in {city}: {response.text}" if response.status_code == 200 else "Failed to fetch weather."

# Tool 3: Get Country Information (REST API)
def get_country_info(country):
    """Fetch general information about a country."""
    url = f"https://restcountries.com/v3.1/name/{country}"
    response = requests.get(url).json()
    
    if isinstance(response, list) and len(response) > 0:
        data = response[0]
        name = data.get("name", {}).get("common", "Unknown")
        capital = data.get("capital", ["Unknown"])[0]
        population = data.get("population", "Unknown")
        currency = list(data.get("currencies", {}).keys())[0] if "currencies" in data else "Unknown"
        return f"Country: {name}, Capital: {capital}, Population: {population}, Currency: {currency}"
    return "Failed to fetch country information."

# Gemini-based Reasoning Function
def art_reasoning(query):
    """Uses Gemini LLM to decide which tool to use based on query."""
    prompt = f"""
    You are an AI assistant that decides the best tool to use based on the user's query.
    
    Available Tools:
    1. Solve Equations (if the query is a mathematical equation).
    2. Fetch Weather (if the query is about weather in a city).
    3. Get Country Information (if the query asks about a country).

    Query: "{query}"
    Analyze the query and return one of the following:
    - 'EQUATION' if it's a math problem.
    - 'WEATHER' if it's asking about weather.
    - 'COUNTRY' if it's asking about a country.
    - 'UNKNOWN' if it doesn't match any category.
    """

    # Ask Gemini LLM to classify the query
    response = gemini_model.generate_content(prompt)
    decision = response.text.strip().upper()

    # Call the appropriate tool
    if "EQUATION" in decision:
        equation = query.split("solve")[-1].strip()
        return solve_equation(equation)
    
    elif "WEATHER" in decision:
        city = query.split("weather in")[-1].strip()
        return get_weather(city)
    
    elif "COUNTRY" in decision:
        country = query.split("about")[-1].strip()
        return get_country_info(country)
    
    else:
        return "I can solve equations, fetch weather, or get country info. Please ask accordingly."

# Example Queries
queries = [
    "solve x**2 - 4 = 0",
    "What is the weather in London?",
    "Tell me about France",
    "Who is the president of the USA?"  # Will return 'UNKNOWN'
]

for query in queries:
    print(f"\nQuery: {query}")
    print(f"Answer: {art_reasoning(query)}")
