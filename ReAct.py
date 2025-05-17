
# import libraries
import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from googlesearch import search
from dotenv import load_dotenv
import argparse
load_dotenv()



# load API keys; you will need to obtain these if you haven't yet
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Initialize Gemini model
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define a search function with content extraction
def google_search_with_content(query, num_results=5):
    search_results = list(search(query, num_results=num_results))  # Get search URLs
    extracted_content = []

    for url in search_results:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}  # Avoid bot detection
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()  # Check if request was successful
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract text content (Modify based on website structure)
            paragraphs = soup.find_all("p")  
            page_content = " ".join([p.get_text() for p in paragraphs[:10]])  # Get first 10 paragraphs
            
            extracted_content.append({"url": url, "content": page_content})
        
        except Exception as e:
            extracted_content.append({"url": url, "content": f"Error fetching content: {e}"})

    return extracted_content

# Wrap function as a LangChain Tool
search_tool = Tool(
    name="Google Search with Content Extraction",
    func=google_search_with_content,
    description="Use this tool to search Google and extract content from top results."
)
def main():
    parser = argparse.ArgumentParser(description='AI Code Automation Agent')
    # parser.add_argument('file_path', help='Path to the file to modify')
    parser.add_argument('stock_name', help='Stock name for processing')
    args = parser.parse_args()
    stock_name = args.stock_name
# Initialize ReAct agent
    agent = initialize_agent(
        tools=[search_tool], llm=llm, agent="zero-shot-react-description", verbose=True)


    # Example query
    response = agent.run("Analyse the stock"+stock_name+"  from multiple trusted sources and provide me market sentiment to buy,sell or hold stock.")
    
    print(response)


    

if __name__ == "__main__":
    main()