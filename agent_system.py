import sys
import os
import logging
import json
import re
from typing import Dict, Tuple, Any
import datetime
from process_booking_functions import process_movie_booking, process_restaurant_booking
    
# Environment Configuration
from dotenv import load_dotenv
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Import Handling
try:
    from pydantic import field_validator, ConfigDict, BaseModel, SkipValidation
except ImportError:
    try:
        from pydantic.v1 import validator as field_validator
        from pydantic.v1 import BaseModel
        ConfigDict = dict
        SkipValidation = None
    except ImportError:
        field_validator = None
        ConfigDict = dict
        BaseModel = object
        SkipValidation = None

# API and External Library Imports
import openai
import requests
import nltk

# CrewAI Imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as CrewAITool

# LangChain Imports
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.tools import BaseTool

# Third-Party Imports
from google.cloud import compute_v1
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from translate import Translator
from crewai_tools import YoutubeChannelSearchTool

# Install NLTK stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# API Key Retrieval
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Validate API Keys
if not OPENAI_API_KEY:
    logger.warning("Missing OpenAI API Key! Set OPENAI_API_KEY in your .env file for full functionality.")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Tool Creation Function
def create_crewai_tool(name, description, func):
    """
    Create a CrewAI tool with a docstring to prevent validation errors
    
    Args:
        name (str): Name of the tool
        description (str): Description of the tool
        func (callable): Function to be used by the tool
    
    Returns:
        CrewAI Tool
    """
    @CrewAITool(name)
    def wrapper(query):
        """
        A dynamically created tool for multi-agent system.
        
        Args:
            query: Input query to process
        
        Returns:
            Result of the tool's function
        """
        return func(query)
    
    return wrapper

# Load environment variables
load_dotenv()

# Retrieve API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_response(prompt: str) -> str:
    """Generate response using OpenAI GPT model."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content


# Utility Functions
def get_weather(city):
    """Fetch weather information using OpenWeather API"""
    if not OPENWEATHER_API_KEY:
        return "Weather Agent: Error: Missing OpenWeather API key."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"Weather Agent: Weather in {city}: {description}, {temperature}°C"
        else:
            return f"Weather Agent: Error fetching weather data ({data.get('message', 'Unknown error')})"
    except Exception as e:
        return f"Weather Agent: {str(e)}"

def get_news(category=None):
    """Fetch top headlines using NewsAPI with optional category filtering"""
    if not NEWS_API_KEY:
        return "News Agent: Error: Missing News API key."

    url = f"https://newsapi.org/v2/top-headlines?country=us"
    
    if category:
        url += f"&category={category}"
    
    url += f"&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            articles = data.get("articles", [])
            if articles:
                headlines = [article["title"] for article in articles[:5]]
                category_display = f" ({category})" if category else ""
                return f"News Agent{category_display}:\n" + "\n".join(headlines)
            else:
                return f"News Agent: No headlines found{' for ' + category if category else ''}."
        else:
            return f"News Agent: Error fetching news ({data.get('message', 'Unknown error')})"
    except Exception as e:
        return f"News Agent: {str(e)}"

def calculate_expression(expression):
    """Math Agent to evaluate expressions"""
    try:
        # Add basic security check to prevent code execution
        allowed_chars = set("0123456789+-*/().^ ")
        for char in expression:
            if char not in allowed_chars:
                return f"Math Agent: Error: Invalid character in expression: '{char}'"
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        result = eval(expression)
        return f"Math Agent: The result of {expression} is {result}"
    except Exception as e:
        return f"Math Agent: Error evaluating expression: {str(e)}"

def generate_code(prompt):
    """Code generation using OpenAI API"""
    if not OPENAI_API_KEY:
        return "Code Agent: Error: Missing OpenAI API key."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled programmer. Generate code based on the user's requirements. Provide only the code with minimal explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return "Code Agent:\n" + response.choices[0].message.content
    except Exception as e:
        return f"Code Agent: Error generating code: {str(e)}"

def get_dictionary_word(word):
    """Fetch word definition dynamically from an API"""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list):
            definition = data[0]["meanings"][0]["definitions"][0]["definition"]
            return f"Dictionary Agent: {word} - {definition}"
        else:
            return "Dictionary Agent: Word not found."
    except Exception as e:
        return f"Dictionary Agent: Error fetching definition: {str(e)}"

def calculate_bmi(weight, height):
    """BMI Calculator Agent"""
    try:
        # Convert to float in case they are strings
        weight = float(weight)
        height = float(height)
        
        if weight <= 0 or height <= 0:
            return "BMI Agent: Error: Weight and height must be positive values."
            
        bmi = weight / (height ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obese"
        
        return f"BMI Agent: Your BMI is {bmi:.2f} ({category})"
    except ZeroDivisionError:
        return "BMI Agent: Error: Height cannot be zero."
    except ValueError:
        return "BMI Agent: Error: Weight and height must be numeric values."
    except Exception as e:
        return f"BMI Agent: Error calculating BMI: {str(e)}"

# Custom search tool using DuckDuckGo
class CustomSearchTool:
    def __init__(self):
        self.search_engine = DuckDuckGoSearchRun()

    def run(self, query: str) -> str:
        return self.search_engine.run(query)

class SearchBaseTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for searching information on the internet"
    
    def _run(self, query):
        return CustomSearchTool().run(query)
        
    async def _arun(self, query):
        return self._run(query)
    
search_tool = SearchBaseTool()

# ✅ Utility Functions

def get_weather(city):
    """Fetch weather information using OpenWeather API"""
    if not OPENWEATHER_API_KEY:
        return "Weather Agent: Error: Missing OpenWeather API key."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"Weather Agent: Weather in {city}: {description}, {temperature}°C"
        else:
            return f"Weather Agent: Error fetching weather data ({data.get('message', 'Unknown error')})"
    except Exception as e:
        return f"Weather Agent: {str(e)}"

def get_news(category=None):
    """Fetch top headlines using NewsAPI with optional category filtering"""
    if not NEWS_API_KEY:
        return "News Agent: Error: Missing News API key."

    # Base URL for the API request
    url = f"https://newsapi.org/v2/top-headlines?country=us"
    
    # Add category parameter if specified
    if category:
        url += f"&category={category}"
    
    # Add API key
    url += f"&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            articles = data.get("articles", [])
            if articles:
                headlines = [article["title"] for article in articles[:5]]
                category_display = f" ({category})" if category else ""
                return f"News Agent{category_display}:\n" + "\n".join(headlines)
            else:
                return f"News Agent: No headlines found{' for ' + category if category else ''}."
        else:
            return f"News Agent: Error fetching news ({data.get('message', 'Unknown error')})"
    except Exception as e:
        return f"News Agent: {str(e)}"

def calculate_expression(expression):
    """Math Agent to evaluate expressions"""
    try:
        # Add basic security check to prevent code execution
        allowed_chars = set("0123456789+-*/().^ ")
        for char in expression:
            if char not in allowed_chars:
                return f"Math Agent: Error: Invalid character in expression: '{char}'"
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        result = eval(expression)
        return f"Math Agent: The result of {expression} is {result}"
    except Exception as e:
        return f"Math Agent: Error evaluating expression: {str(e)}"

def generate_code(prompt):
    """Code generation using OpenAI API"""
    if not OPENAI_API_KEY:
        return "Code Agent: Error: Missing OpenAI API key."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled programmer. Generate code based on the user's requirements. Provide only the code with minimal explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return "Code Agent:\n" + response.choices[0].message.content
    except Exception as e:
        return f"Code Agent: Error generating code: {str(e)}"

def get_dictionary_word(word):
    """Fetch word definition dynamically from an API"""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list):
            definition = data[0]["meanings"][0]["definitions"][0]["definition"]
            return f"Dictionary Agent: {word} - {definition}"
        else:
            return "Dictionary Agent: Word not found."
    except Exception as e:
        return f"Dictionary Agent: Error fetching definition: {str(e)}"

def calculate_bmi(weight, height):
    """BMI Calculator Agent"""
    try:
        # Convert to float in case they are strings
        weight = float(weight)
        height = float(height)
        
        if weight <= 0 or height <= 0:
            return "BMI Agent: Error: Weight and height must be positive values."
            
        bmi = weight / (height ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obese"
        
        return f"BMI Agent: Your BMI is {bmi:.2f} ({category})"
    except ZeroDivisionError:
        return "BMI Agent: Error: Height cannot be zero."
    except ValueError:
        return "BMI Agent: Error: Weight and height must be numeric values."
    except Exception as e:
        return f"BMI Agent: Error calculating BMI: {str(e)}"

# ✅ Movie Booking Integration Function
def process_movie_booking(user_prompt):
    """Process a movie booking request"""
        
    logging.info(f"Processing movie booking request: {user_prompt}")
    try:
        # Import directly from process_booking_functions.py
        from process_booking_functions import process_movie_booking as movie_booking_handler
        result = movie_booking_handler(user_prompt)
        return result
    except Exception as e:
        logging.error(f"Error in movie booking: {str(e)}")
        return f"Movie Booking Agent: Error processing your request: {str(e)}"

# ✅ Restaurant Booking Integration Function
def process_restaurant_booking(user_prompt):
    """Process a restaurant booking request"""
        
    logging.info(f"Processing restaurant booking request: {user_prompt}")
    try:
        # Import directly from process_booking_functions.py
        from process_booking_functions import process_restaurant_booking as restaurant_booking_handler
        result = restaurant_booking_handler(user_prompt)
        return result
    except Exception as e:
        logging.error(f"Error in restaurant booking: {str(e)}")
        return f"Restaurant Booking Agent: Error processing your request: {str(e)}"

# -----------------------------------------------------------------------------
# New Utility Classes for Added Agents
# -----------------------------------------------------------------------------

# GCP VM Tool Classes
class GCPVMTool:
    def __init__(self, project_id, zone):
        self.project_id = project_id
        self.zone = zone
        self.instance_client = compute_v1.InstancesClient()
    
    def create_vm(self, instance_name, machine_type, source_image, network="global/networks/default"):
        """
        Create a VM instance on GCP.
        """
        instance = compute_v1.Instance()
        instance.name = instance_name
        instance.machine_type = f"zones/{self.zone}/machineTypes/{machine_type}"
        
        # Configure the disk
        disk = compute_v1.AttachedDisk()
        disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
            source_image=source_image
        )
        disk.auto_delete = True
        disk.boot = True
        instance.disks = [disk]
        
        # Configure the network interface
        network_interface = compute_v1.NetworkInterface()
        network_interface.network = f"projects/{self.project_id}/global/networks/default"
        instance.network_interfaces = [network_interface]
        
        # Create the VM
        operation = self.instance_client.insert(
            project=self.project_id,
            zone=self.zone,
            instance_resource=instance
        )
        
        return f"VM creation started. Operation ID: {operation.name}"

# GCP Instance Tool Class
class GCPInstanceTool:
    def __init__(self, project_id: str, zone: str):
        self.project_id = project_id
        self.zone = zone
        self.instance_template_client = compute_v1.InstanceTemplatesClient()
        self.instance_group_manager_client = compute_v1.InstanceGroupManagersClient()
        self.autoscaler_client = compute_v1.AutoscalersClient()
        self.instances_client = compute_v1.InstancesClient()

    def create_instance_template_from_existing_vm(
        self,
        template_name: str,
        source_vm_name: str,
        source_image: str = "projects/debian-cloud/global/images/family/debian-11",
    ):
        """Create an instance template from an existing VM."""
        # Get the source VM
        source_vm = self.instances_client.get(
            project=self.project_id,
            zone=self.zone,
            instance=source_vm_name,
        )

        # Extract the machine type name from the URL
        machine_type_url = source_vm.machine_type
        machine_type_name = machine_type_url.split("/")[-1]

        # Define the disk configuration for the instance template
        disk = compute_v1.AttachedDisk(
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=source_image
            ),
            auto_delete=True,
            boot=True,
        )

        # Define the instance template
        template = compute_v1.InstanceTemplate()
        template.name = template_name
        template.properties = compute_v1.InstanceProperties(
            machine_type=machine_type_name,
            disks=[disk],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    name=source_vm.network_interfaces[0].name
                )
            ],
        )

        # Create the instance template
        operation = self.instance_template_client.insert(
            project=self.project_id,
            instance_template_resource=template,
        )
        return template_name

    def create_instance_group(
        self,
        group_name: str,
        template_name: str,
        base_instance_name: str,
        target_size: int = 1,
    ):
        """Create a managed instance group."""
        # Define the instance group manager
        instance_group_manager = compute_v1.InstanceGroupManager(
            name=group_name,
            base_instance_name=base_instance_name,
            instance_template=f"projects/{self.project_id}/global/instanceTemplates/{template_name}",
            target_size=target_size,
        )

        # Create the instance group
        operation = self.instance_group_manager_client.insert(
            project=self.project_id,
            zone=self.zone,
            instance_group_manager_resource=instance_group_manager,
        )
        return group_name

    def create_autoscaling_policy(
        self,
        instance_group_name: str,
        target_cpu_utilization: float = 0.7,
        min_instances: int = 1,
        max_instances: int = 5,
    ):
        """Create an autoscaling policy for a managed instance group."""
        # Define the autoscaler
        autoscaler = compute_v1.Autoscaler(
            name=f"{instance_group_name}-autoscaler",
            target=f"projects/{self.project_id}/zones/{self.zone}/instanceGroupManagers/{instance_group_name}",
            autoscaling_policy=compute_v1.AutoscalingPolicy(
                cpu_utilization=compute_v1.AutoscalingPolicyCpuUtilization(
                    utilization_target=target_cpu_utilization
                ),
                min_num_replicas=min_instances,
                max_num_replicas=max_instances,
            ),
        )

        # Create the autoscaler
        operation = self.autoscaler_client.insert(
            project=self.project_id,
            zone=self.zone,
            autoscaler_resource=autoscaler,
        )

        # Wait for the operation to complete
        operation.result()

        return f"Autoscaling policy created. Operation ID: {operation.name}"

# Text Similarity Class
class TextSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
    
    def calculate_similarity(self, text1, text2):
        """Calculate the similarity between two texts using Sentence Transformers."""
        try:
            # Encode the texts
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            return similarity.item()
        except Exception as e:
            return f"Error calculating similarity: {e}"

# Web Search Class
class WebSearch:
    def __init__(self, api_key=None):
        self.api_key = api_key  # API key for SerpAPI (optional)
    
    def search_google(self, query, num_results=5):
        """Search Google for the given query and return the top results."""
        if self.api_key:
            # Use SerpAPI for more reliable and structured results
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])
            return [result.get("snippet", "") for result in results]
        else:
            # Fallback to simple web scraping (less reliable)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            url = f"https://www.google.com/search?q={query}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for g in soup.find_all("div", class_="tF2Cxc"):
                snippet = g.find("div", class_="IsZvec")
                if snippet:
                    results.append(snippet.text)
                if len(results) >= num_results:
                    break
            return results

class TranslatorUtil:
    def __init__(self):
        pass
    
    def translate_text(self, text, src_lang='auto', dest_lang='en'):
        """Translate text from one language to another."""
        try:
            # Initialize translator with target language
            translator = Translator(to_lang=dest_lang)
            if src_lang != 'auto':
                translator = Translator(from_lang=src_lang, to_lang=dest_lang)
            
            # For longer texts, break into chunks to avoid length limitations
            max_chunk_size = 500
            if len(text) <= max_chunk_size:
                return translator.translate(text)
            else:
                # Break text into sentences or chunks and translate each
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                translated_chunks = [translator.translate(chunk) for chunk in chunks]
                return ' '.join(translated_chunks)
        except Exception as e:
            return f"Translation failed: {e}"

# YouTube Tool Initializer
def initialize_yt_tool(youtube_channel_handle='@default'):
    """Initialize YouTube tool with proper format for CrewAI Agent"""
    # Create a crewai-compatible Tool instead of using YoutubeChannelSearchTool directly
    yt_search_tool = YoutubeChannelSearchTool(youtube_channel_handle=youtube_channel_handle)
    
    # Convert to a Tool that CrewAI can use
    from crewai import Tool as CrewAITool
    
    return CrewAITool(
        name="YouTube Channel Search",
        description="Search for videos in a specific YouTube channel",
        func=lambda query: yt_search_tool.run(query)
    )

# Functions for the new integrated agents
def process_translation(text, src_lang='auto', dest_lang='en'):
    """Process a translation request"""
    logging.info(f"Processing translation request: {text} from {src_lang} to {dest_lang}")
    try:
        translator_util = TranslatorUtil()
        result = translator_util.translate_text(text, src_lang, dest_lang)
        return f"Translation Agent: {result}"
    except Exception as e:
        logging.error(f"Error in translation: {str(e)}")
        return f"Translation Agent: Error: {str(e)}"

def process_plagiarism_detection(text, threshold=0.5, num_results=5):
    """Process a plagiarism detection request"""
    logging.info(f"Processing plagiarism detection for text: {text[:50]}...")
    try:
        text_similarity = TextSimilarity()
        web_search = WebSearch(api_key=SERPAPI_KEY)
        
        # Extract keywords or use a more concise query
        query = " ".join(text.split()[:10])  # Use the first 10 words as the search query
        
        # Search the internet for similar content
        reference_texts = web_search.search_google(query, num_results=num_results)
        
        # Compare the text with search results
        results = []
        for ref_text in reference_texts:
            similarity = text_similarity.calculate_similarity(text, ref_text)
            if similarity >= threshold:
                results.append((ref_text, similarity))
        
        # Format the response
        if results:
            response = "Plagiarism Detection Agent:\n"
            for i, (ref_text, similarity) in enumerate(results, 1):
                response += f"{i}. Similarity: {similarity:.2f}\n   Text: {ref_text[:100]}...\n\n"
            return response
        else:
            return "Plagiarism Detection Agent: No significant similarities found."
    except Exception as e:
        logging.error(f"Error in plagiarism detection: {str(e)}")
        return f"Plagiarism Detection Agent: Error: {str(e)}"

def process_gcp_vm_creation(project_id, zone, instance_name, machine_type, source_image):
    """Process a GCP VM creation request"""

    # Use default values if parameters are None
    project_id = project_id or "tokyo-mark-452209-h9"
    zone = zone or "us-central1-a"
    instance_name = instance_name or "default-instance"
    machine_type = machine_type or "e2-medium"
    source_image = source_image or "projects/debian-cloud/global/images/family/debian-11"
    
    
    logging.info(f"Processing GCP VM creation: {instance_name} in {zone}")
    try:
        gcp_tool = GCPVMTool(project_id=project_id, zone=zone)
        result = gcp_tool.create_vm(
            instance_name=instance_name,
            machine_type=machine_type,
            source_image=source_image
        )
        return f"GCP VM Agent: {result}"
    except Exception as e:
        logging.error(f"Error in GCP VM creation: {str(e)}")
        return f"GCP VM Agent: Error: {str(e)}"

def process_gcp_autoscaling(project_id, zone, template_name, source_vm_name, group_name, base_instance_name):
    """Process a GCP autoscaling setup request"""
        # Use default values if parameters are None
    project_id = project_id or "tokyo-mark-452209-h9"
    zone = zone or "us-central1-a"
    
    # Generate default names based on timestamp if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    template_name = template_name or f"template-{timestamp}"
    source_vm_name = source_vm_name or "default-vm"  # This should be an existing VM
    group_name = group_name or f"group-{timestamp}"
    base_instance_name = base_instance_name or f"instance-{timestamp}"
    
    logging.info(f"Processing GCP autoscaling setup for group: {group_name}")
    try:
        # First check if the source VM exists
        try:
            instances_client = compute_v1.InstancesClient()
            instances_client.get(
                project=project_id,
                zone=zone,
                instance=source_vm_name
            )
        except Exception as vm_error:
            # Source VM not found, return informative error
            error_msg = f"Source VM '{source_vm_name}' not found in project '{project_id}', zone '{zone}'. Please specify an existing VM or create one first."
            logging.error(error_msg)
            return f"GCP Autoscaling Agent: Error: {error_msg}"
            
        gcp_tool = GCPInstanceTool(project_id=project_id, zone=zone)
        
        # Step 1: Create instance template from existing VM
        template_result = gcp_tool.create_instance_template_from_existing_vm(
            template_name=template_name,
            source_vm_name=source_vm_name
        )
        
        # Step 2: Create instance group
        group_result = gcp_tool.create_instance_group(
            group_name=group_name,
            template_name=template_name,
            base_instance_name=base_instance_name
        )
        
        # Step 3: Set up autoscaling policy
        policy_result = gcp_tool.create_autoscaling_policy(
            instance_group_name=group_name
        )

    
        
        return f"GCP Autoscaling Agent: Template created: {template_result}, Group created: {group_result}, {policy_result}"
    except Exception as e:
        logging.error(f"Error in GCP autoscaling setup: {str(e)}")
        return f"GCP Autoscaling Agent: Error: {str(e)}"

def process_youtube_summary(channel_handle, topic):
    """Process a YouTube video summary request"""
    logging.info(f"Processing YouTube summary for channel: {channel_handle} on topic: {topic}")
    try:
        # Initialize the YouTube tool and fetch a video
        if not OPENAI_API_KEY:
            return "YouTube Summary Agent: Error: Missing OpenAI API key."
            
        # Create a direct instance of YoutubeChannelSearchTool
        yt_search_tool = YoutubeChannelSearchTool(youtube_channel_handle=channel_handle)
        
        # Use the _run method with the search_query parameter
        search_result = yt_search_tool._run(search_query=topic)  # Changed from run() to _run(search_query=topic)
        
        if not search_result:
            return f"YouTube Summary Agent: No videos found for topic '{topic}' on channel @{channel_handle}"
        
        # Generate a summary using OpenAI
        prompt = f"""
        Summarize the following YouTube video information about {topic}:
        {search_result}
        
        Provide a concise 3-paragraph summary covering:
        1. The main topic and key points
        2. Important details or examples mentioned
        3. Conclusions or takeaways
        """
        
        summary = get_response(prompt)
        return f"YouTube Summary Agent:\n{summary}"
    except Exception as e:
        logging.error(f"Error in YouTube summary: {str(e)}")
        return f"YouTube Summary Agent: Error: {str(e)}"
def process_stock_info(ticker: str):
    """Fetch current stock information"""
    logging.info(f"Fetching stock info for: {ticker}")
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker.upper()}&token={os.getenv('FINNHUB_API_KEY')}"
        response = requests.get(url)
        data = response.json()
        if 'c' in data:
            return f"Stock Agent: {ticker.upper()} is currently trading at ${data['c']}."
        return "Stock Agent: Could not retrieve stock data."
    except Exception as e:
        return f"Stock Agent: Error: {str(e)}"

def process_recipe_request(query: str):
    """Fetch recipe based on ingredients or meal type"""
    logging.info(f"Fetching recipe for: {query}")
    try:
        url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&number=1&apiKey={os.getenv('SPOONACULAR_API_KEY')}"
        response = requests.get(url)
        data = response.json()
        if data["results"]:
            title = data["results"][0]["title"]
            return f"Recipe Agent: Here's a recipe suggestion: {title}"
        return "Recipe Agent: No recipes found."
    except Exception as e:
        return f"Recipe Agent: Error: {str(e)}"

def process_idea_generation(prompt: str):
    """Generate creative ideas for a topic."""
    logging.info(f"Generating ideas for: {prompt}")
    try:
        idea_prompt = f"Generate 5 creative and original ideas related to: {prompt}"
        return f"Idea Generator Agent:\n{get_response(idea_prompt)}"
    except Exception as e:
        return f"Idea Generator Agent: Error: {str(e)}"

def process_meditation_prompt(prompt: str):
    """Generate a calming meditation script or guidance."""
    logging.info(f"Creating meditation for: {prompt}")
    try:
        med_prompt = f"Guide a user through a {prompt}-themed meditation lasting around 5 minutes."
        return f"Meditation Agent:\n{get_response(med_prompt)}"
    except Exception as e:
        return f"Meditation Agent: Error: {str(e)}"

def process_book_recommendation(topic: str):
    """Recommend books based on user interest."""
    logging.info(f"Book recommendation for: {topic}")
    try:
        prompt = f"Suggest 5 book recommendations based on: {topic}. Include title and author."
        return f"Book Recommender Agent:\n{get_response(prompt)}"
    except Exception as e:
        return f"Book Recommender Agent: Error: {str(e)}"

def process_unit_conversion(expression: str):
    """Convert units using LLM if API isn't available."""
    logging.info(f"Converting units: {expression}")
    try:
        prompt = f"Convert the following units: {expression}. Provide only the numeric result with unit."
        return f"Unit Converter Agent:\n{get_response(prompt)}"
    except Exception as e:
        return f"Unit Converter Agent: Error: {str(e)}"

def process_quote(topic: str):
    """Fetch a motivational/inspirational quote."""
    logging.info(f"Fetching quote for: {topic}")
    try:
        quote_prompt = f"Give me an inspirational or philosophical quote about: {topic}. Mention the author."
        return f"Quote Agent:\n{get_response(quote_prompt)}"
    except Exception as e:
        return f"Quote Agent: Error: {str(e)}"

def process_movie_recommendation(criteria: str):
    """Recommend movies by genre, mood, or keywords."""
    logging.info(f"Recommending movie for: {criteria}")
    try:
        movie_prompt = f"Recommend 5 movies based on this criteria: {criteria}. Include title and genre."
        return f"Movie Recommender Agent:\n{get_response(movie_prompt)}"
    except Exception as e:
        return f"Movie Recommender Agent: Error: {str(e)}"

def process_skincare_advice(query: str):
    logging.info(f"Processing skincare query: {query}")
    try:
        prompt = f"Suggest a skincare routine and product tips for: {query}."
        return f"Skincare Agent:\n{get_response(prompt)}"
    except Exception as e:
        return f"Skincare Agent: Error: {str(e)}"

def process_haircare_advice(query: str):
    logging.info(f"Processing haircare query: {query}")
    try:
        prompt = f"Suggest a haircare routine and tips for: {query}."
        return f"Haircare Agent:\n{get_response(prompt)}"
    except Exception as e:
        return f"Haircare Agent: Error: {str(e)}"

def process_mythology_info(civilization: str, query: str):
    logging.info(f"Mythology request for {civilization}: {query}")
    try:
        prompt = f"Explain the mythology of {civilization}. Focus on: {query}."
        return f"Mythology Agent ({civilization}):\n{get_response(prompt)}"
    except Exception as e:
        return f"Mythology Agent ({civilization}): Error: {str(e)}"
    
def process_fitness_coach(prompt: str):
    logging.info(f"Processing fitness advice for: {prompt}")
    try:
        prompt_text = f"Create a customized fitness plan or tips for: {prompt}"
        return f"Fitness Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Fitness Coach Agent: Error: {str(e)}"

def process_meal_planner(prompt: str):
    logging.info(f"Processing meal plan for: {prompt}")
    try:
        prompt_text = f"Generate a weekly meal plan for: {prompt}"
        return f"Meal Planner Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Meal Planner Agent: Error: {str(e)}"

def process_financial_advice(prompt: str):
    logging.info(f"Processing financial advice for: {prompt}")
    try:
        prompt_text = f"Provide basic personal finance advice for: {prompt}"
        return f"Financial Advisor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Financial Advisor Agent: Error: {str(e)}"

def process_travel_guide(prompt: str):
    logging.info(f"Processing travel guide for: {prompt}")
    try:
        prompt_text = f"Recommend destinations and travel tips for: {prompt}"
        return f"Travel Guide Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Travel Guide Agent: Error: {str(e)}"

def process_language_tutor(prompt: str):
    logging.info(f"Processing language help for: {prompt}")
    try:
        prompt_text = f"Help a user learn a new language. Topic: {prompt}"
        return f"Language Tutor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Language Tutor Agent: Error: {str(e)}"

def process_dream_interpretation(prompt: str):
    logging.info(f"Interpreting dream: {prompt}")
    try:
        prompt_text = f"Interpret this dream and its symbolism: {prompt}"
        return f"Dream Interpreter Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Dream Interpreter Agent: Error: {str(e)}"

def process_relationship_advice(prompt: str):
    logging.info(f"Relationship advice for: {prompt}")
    try:
        prompt_text = f"Offer relationship advice based on this: {prompt}"
        return f"Relationship Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Relationship Coach Agent: Error: {str(e)}"

def process_job_finder(prompt: str):
    logging.info(f"Suggesting job roles for: {prompt}")
    try:
        prompt_text = f"Suggest job roles and career paths based on: {prompt}"
        return f"Job Finder Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Job Finder Agent: Error: {str(e)}"

def process_gardening_advice(prompt: str):
    logging.info(f"Gardening advice for: {prompt}")
    try:
        prompt_text = f"Provide gardening tips for: {prompt}"
        return f"Gardening Expert Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Gardening Expert Agent: Error: {str(e)}"

def process_parenting_tips(prompt: str):
    logging.info(f"Parenting tips for: {prompt}")
    try:
        prompt_text = f"Give parenting advice for: {prompt}"
        return f"Parenting Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Parenting Coach Agent: Error: {str(e)}"

def process_resume_review(prompt: str):
    logging.info(f"Resume review for: {prompt}")
    try:
        prompt_text = f"Review and improve this resume detail: {prompt}"
        return f"Resume Reviewer Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Resume Reviewer Agent: Error: {str(e)}"

def process_horoscope(prompt: str):
    logging.info(f"Horoscope request for: {prompt}")
    try:
        prompt_text = f"Give a horoscope reading for: {prompt}"
        return f"Horoscope Reader Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Horoscope Reader Agent: Error: {str(e)}"

def process_event_planner(prompt: str):
    logging.info(f"Event planning for: {prompt}")
    try:
        prompt_text = f"Help organize an event: {prompt}"
        return f"Event Planner Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Event Planner Agent: Error: {str(e)}"

def process_shopping(prompt: str):
    logging.info(f"Shopping help for: {prompt}")
    try:
        prompt_text = f"Help find deals or product suggestions for: {prompt}"
        return f"Shopping Assistant Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Shopping Assistant Agent: Error: {str(e)}"

def process_mental_health(prompt: str):
    logging.info(f"Mental health support for: {prompt}")
    try:
        prompt_text = f"Give mindfulness or stress relief tips for: {prompt}"
        return f"Mental Health Support Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Mental Health Support Agent: Error: {str(e)}"

def process_productivity(prompt: str):
    logging.info(f"Productivity boost for: {prompt}")
    try:
        prompt_text = f"Suggest productivity tools or strategies for: {prompt}"
        return f"Productivity Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Productivity Coach Agent: Error: {str(e)}"

def process_budget_tracker(prompt: str):
    logging.info(f"Budget tracking for: {prompt}")
    try:
        prompt_text = f"Help track and optimize expenses for: {prompt}"
        return f"Budget Tracker Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Budget Tracker Agent: Error: {str(e)}"

def process_affirmations(prompt: str):
    logging.info(f"Affirmations for: {prompt}")
    try:
        prompt_text = f"Generate daily affirmations for: {prompt}"
        return f"Affirmation Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Affirmation Agent: Error: {str(e)}"

def process_nutrition(prompt: str):
    logging.info(f"Nutrition tips for: {prompt}")
    try:
        prompt_text = f"Suggest balanced diet ideas for: {prompt}"
        return f"Nutritionist Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Nutritionist Agent: Error: {str(e)}"

def process_pet_care(prompt: str):
    logging.info(f"Pet care for: {prompt}")
    try:
        prompt_text = f"Give pet health or training advice for: {prompt}"
        return f"Pet Care Advisor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Pet Care Advisor Agent: Error: {str(e)}"

def process_home_decor(prompt: str):
    logging.info(f"Home decor for: {prompt}")
    try:
        prompt_text = f"Suggest interior design ideas for: {prompt}"
        return f"Home Decor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Home Decor Agent: Error: {str(e)}"

def process_sleep(prompt: str):
    logging.info(f"Sleep tips for: {prompt}")
    try:
        prompt_text = f"Give sleep improvement tips for: {prompt}"
        return f"Sleep Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Sleep Coach Agent: Error: {str(e)}"

def process_tech_support(prompt: str):
    logging.info(f"Tech issue for: {prompt}")
    try:
        prompt_text = f"Help troubleshoot this issue: {prompt}"
        return f"Tech Support Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Tech Support Agent: Error: {str(e)}"

def process_time_management(prompt: str):
    logging.info(f"Time management for: {prompt}")
    try:
        prompt_text = f"Help schedule and prioritize tasks: {prompt}"
        return f"Time Management Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Time Management Agent: Error: {str(e)}"

def process_career_counselor(prompt: str):
    logging.info(f"Career guidance for: {prompt}")
    try:
        prompt_text = f"Give career growth or switch advice for: {prompt}"
        return f"Career Counselor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Career Counselor Agent: Error: {str(e)}"

def process_academic_tutor(prompt: str):
    logging.info(f"Academic help for: {prompt}")
    try:
        prompt_text = f"Explain this academic concept: {prompt}"
        return f"Academic Tutor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Academic Tutor Agent: Error: {str(e)}"

def process_festival_finder(prompt: str):
    logging.info(f"Festival info for: {prompt}")
    try:
        prompt_text = f"Find festivals related to: {prompt}"
        return f"Festival Finder Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Festival Finder Agent: Error: {str(e)}"

def process_resume_keywords(prompt: str):
    logging.info(f"Optimizing resume for keywords: {prompt}")
    try:
        prompt_text = f"Add ATS-friendly keywords for: {prompt}"
        return f"Resume Keyword Optimizer Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Resume Keyword Optimizer Agent: Error: {str(e)}"

def process_cover_letter(prompt: str):
    logging.info(f"Cover letter for: {prompt}")
    try:
        prompt_text = f"Create a professional cover letter for: {prompt}"
        return f"Cover Letter Builder Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Cover Letter Builder Agent: Error: {str(e)}"

def process_voice_tone(prompt: str):
    logging.info(f"Voice tone coaching for: {prompt}")
    try:
        prompt_text = f"Suggest improvements to speaking tone for: {prompt}"
        return f"Voice Tone Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Voice Tone Coach Agent: Error: {str(e)}"

def process_crypto(prompt: str):
    logging.info(f"Crypto advice for: {prompt}")
    try:
        prompt_text = f"Explain this crypto concept: {prompt}"
        return f"Crypto Advisor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Crypto Advisor Agent: Error: {str(e)}"

def process_loan_calculator(prompt: str):
    logging.info(f"Loan/EMI calculation for: {prompt}")
    try:
        prompt_text = f"Calculate loan or EMI for: {prompt}"
        return f"Loan & EMI Calculator Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Loan & EMI Calculator Agent: Error: {str(e)}"

def process_insurance(prompt: str):
    logging.info(f"Insurance help for: {prompt}")
    try:
        prompt_text = f"Explain insurance options for: {prompt}"
        return f"Insurance Helper Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Insurance Helper Agent: Error: {str(e)}"

def process_car_advice(prompt: str):
    logging.info(f"Car buying/selling for: {prompt}")
    try:
        prompt_text = f"Suggest car buying/selling advice for: {prompt}"
        return f"Car Advisor Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Car Advisor Agent: Error: {str(e)}"

def process_appliance(prompt: str):
    logging.info(f"Appliance suggestion for: {prompt}")
    try:
        prompt_text = f"Recommend appliances for: {prompt}"
        return f"Appliance Recommender Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Appliance Recommender Agent: Error: {str(e)}"

def process_minimalism(prompt: str):
    logging.info(f"Minimalism tips for: {prompt}")
    try:
        prompt_text = f"Help live a simpler, decluttered life focused on: {prompt}"
        return f"Minimalism Coach Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Minimalism Coach Agent: Error: {str(e)}"

def process_coding_interview(prompt: str):
    logging.info(f"Interview prep for coding: {prompt}")
    try:
        prompt_text = f"Prepare for a coding interview. Focus: {prompt}"
        return f"Coding Interview Prep Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Coding Interview Prep Agent: Error: {str(e)}"

def process_habits(prompt: str):
    logging.info(f"Tracking habits for: {prompt}")
    try:
        prompt_text = f"Help track and build good habits related to: {prompt}"
        return f"Habit Tracker Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Habit Tracker Agent: Error: {str(e)}"

def process_fashion(prompt: str):
    logging.info(f"Fashion advice for: {prompt}")
    try:
        prompt_text = f"Suggest an outfit or fashion tip for: {prompt}"
        return f"Fashion Stylist Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Fashion Stylist Agent: Error: {str(e)}"

def process_music_recommender(prompt: str):
    logging.info(f"Music recommendation for: {prompt}")
    try:
        prompt_text = f"Recommend music based on: {prompt}"
        return f"Music Recommender Agent:\n{get_response(prompt_text)}"
    except Exception as e:
        return f"Music Recommender Agent: Error: {str(e)}"



# ✅ Agent System with Logging and Parallel Execution
class AgentSystem:
    def __init__(self):
        try:
            
            self.search_tool = SearchBaseTool()

            # Agents from File 1
            self.content_writer = Agent(
                role='Content Writer',
                goal='Create high-quality, engaging content based on given topics.',
                backstory="An experienced content writer with deep research skills.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.resume_builder = Agent(
                role='Resume Builder',
                goal='Create professional resumes tailored to job profiles.',
                backstory="An expert resume writer with ATS system knowledge.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.interview_prep = Agent(
                role='Interview Coach',
                goal='Prepare candidates for job interviews with guidance.',
                backstory="An experienced interview coach for technical and behavioral interviews.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.history_agent = Agent(
                role='History Expert',
                goal='Provide comprehensive historical information and analysis.',
                backstory="A history expert with deep knowledge across various periods.",
                #tools=[self.search_tool],
                verbose=True
            )

            # Utility Agents
            self.weather_agent = Agent(
                role='Weather Expert',
                goal='Provide accurate weather information for locations.',
                backstory="A meteorologist who provides reliable weather forecasts.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.news_agent = Agent(
                role='News Reporter',
                goal='Deliver the latest news headlines.',
                backstory="A journalist who reports on breaking news and current events.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.math_agent = Agent(
                role='Math Wizard',
                goal='Perform mathematical calculations and evaluations.',
                backstory="A mathematics expert who can solve complex problems.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.code_agent = Agent(
                role='Code Generator',
                goal='Generate code based on given requirements.',
                backstory="A skilled programmer who can write efficient and effective code.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.dictionary_agent = Agent(
                role='Language Expert',
                goal='Provide definitions and explanations of words.',
                backstory="A linguist with comprehensive knowledge of language and vocabulary.",
                #tools=[self.search_tool],
                verbose=True
            )

            self.bmi_agent = Agent(
                role='Health Advisor',
                goal='Calculate BMI and provide health insights.',
                backstory="A health expert who helps people understand their physical condition.",
                #tools=[self.search_tool],
                verbose=True
            )
            
            # Movie and restaurant booking agents
            self.movie_agent = Agent(
                role='Movie Ticket Specialist',
                goal='Help users find and book movie tickets based on their preferences.',
                backstory="A movie enthusiast and booking specialist who helps people find the perfect movie experience.",
                #tools=[self.search_tool],
                verbose=True
            )
            
            self.restaurant_agent = Agent(
                role='Restaurant Reservation Specialist',
                goal='Help users find and book restaurant reservations based on their preferences.',
                backstory="A dining expert who helps people find and book the perfect restaurant experience.",
                #tools=[self.search_tool],
                verbose=True
            )
            
            # New integrated agents
            self.translation_agent = Agent(
                role='Translator',
                goal='Translate text from one language to another with high accuracy.',
                backstory="A multilingual expert who can translate between numerous languages fluently.",
                #tools=[self.search_tool],
                verbose=True,
                allow_delegation=False
            )
            
            self.plagiarism_detector = Agent(
                role='Plagiarism Detector',
                goal='Detect plagiarism in a given text by comparing it to online sources.',
                backstory="A vigilant agent trained to identify copied or unoriginal content by searching the internet.",
                #tools=[self.search_tool],
                verbose=True,
                allow_delegation=False
            )
            
            self.gcp_vm_creator = Agent(
                role='GCP VM Creator',
                goal='Create virtual machines on Google Cloud Platform',
                backstory="An AI agent specialized in creating and managing virtual machines on GCP with deep knowledge of GCP's Compute Engine.",
                #tools=[self.search_tool],
                verbose=True,
                allow_delegation=False
            )
            
            self.gcp_autoscaling_manager = Agent(
                role='GCP Autoscaling Manager',
                goal='Set up autoscaling policies for virtual machines on Google Cloud Platform',
                backstory="An AI agent specialized in managing autoscaling policies for GCP Compute Engine that ensures VM instances scale efficiently.",
                #tools=[self.search_tool],
                verbose=True,
                allow_delegation=False
            )
            
            # Create a properly formatted YouTube tool for the agent
            youtube_tool = initialize_yt_tool('@default')  # Use the function you already defined
            
            self.youtube_summarizer = Agent(
                role='YouTube Content Summarizer',
                goal='Summarize YouTube videos on specified topics',
                backstory="An expert who can extract key information from videos and create concise, informative summaries.",
                #tools=[youtube_tool],  # Use the properly formatted tool
                verbose=True,
                allow_delegation=False
                )  
            self.stock_agent = Agent(
            role='Stock Analyst',
            goal='Provide current stock prices and trends.',
            backstory="A financial expert who understands market movements and provides quick price updates.",
            verbose=True
                )

            self.recipe_agent = Agent(
            role='Recipe Recommender',
            goal='Suggest delicious recipes based on ingredients or type of meal.',
            backstory="A passionate chef who recommends recipes from a wide range of cuisines.",
            verbose=True
                )
            
            self.idea_agent = Agent(
            role='Idea Generator',
            goal='Generate creative and original ideas.',
            backstory="A creative strategist who brainstorms fresh ideas.",
            verbose=True
                )

            self.meditation_agent = Agent(
            role='Meditation Coach',
            goal='Guide users through calming meditations.',
            backstory="A calm and wise meditation guide.",
            verbose=True
                )

            self.book_agent = Agent(
            role='Book Recommender',
            goal='Suggest books based on preferences.',
            backstory="A bookworm AI who recommends great reads.",
            verbose=True
                )

            self.unit_converter_agent = Agent(
            role='Unit Converter',
            goal='Convert between metric and imperial units.',
            backstory="An expert in physical measurements and units.",
            verbose=True
                )

            self.quote_agent = Agent(
            role='Quote Collector',
            goal='Deliver inspiring quotes on any topic.',
            backstory="A philosopher with a library of timeless quotes.",
            verbose=True
            )

            self.movie_recommender_agent = Agent(
            role='Movie Buff',
            goal='Suggest movies by mood, genre, or actor.',
            backstory="A cinema lover with a deep movie memory.",
            verbose=True
            )

            self.skincare_agent = Agent(
            role='Skincare Advisor',
            goal='Suggest skincare tips and routines.',
            backstory="A dermatologist AI who helps people glow.",
            verbose=True
            )

            self.haircare_agent = Agent(
            role='Haircare Advisor',
            goal='Advise users on healthy hair routines.',
            backstory="A stylist who loves great hair days.",
            verbose=True
            )

            self.mythology_agent = Agent(
            role='Mythology Expert',
            goal='Explain stories from ancient civilizations.',
            backstory="A myth keeper with timeless knowledge of gods and heroes.",
            verbose=True
            )

            self.fitness_agent = Agent(
            role='Fitness Coach',
            goal='Provide custom workout plans and fitness tips.',
            backstory="A personal trainer with experience in strength, cardio, and flexibility training.",
            verbose=True
            )

            self.meal_agent = Agent(
            role='Meal Planner',
            goal='Suggest weekly meal plans and food prep ideas.',
            backstory="A nutrition-conscious planner with tasty and balanced meal ideas.",
            verbose=True
            )

            self.finance_agent = Agent(
            role='Financial Advisor',
            goal='Offer practical personal finance tips.',
            backstory="A finance guru who simplifies budgeting, saving, and investing for everyone.",
            verbose=True
            )

            self.travel_agent = Agent(
            role='Travel Guide',
            goal='Recommend travel destinations and planning tips.',
            backstory="A seasoned traveler who knows the best places, hacks, and packing tips.",
            verbose=True
            )

            self.language_agent = Agent(
            role='Language Tutor',
            goal='Help users learn new languages efficiently.',
            backstory="A multilingual linguist who teaches grammar, vocabulary, and phrases with ease.",
            verbose=True
            )

            self.dream_agent = Agent(
            role='Dream Interpreter',
            goal='Interpret the meanings behind dreams.',
            backstory="A mystical interpreter with knowledge of dream psychology and symbolism.",
            verbose=True
            )

            self.relationship_agent = Agent(
            role='Relationship Coach',
            goal='Offer relationship and communication advice.',
            backstory="A compassionate coach who supports healthy romantic and social relationships.",
            verbose=True
            )

            self.job_agent = Agent(
            role='Job Finder',
            goal='Suggest career paths or job options.',
            backstory="A career strategist who matches skills to job opportunities.",
            verbose=True
            )

            self.gardening_agent = Agent(
            role='Gardening Expert',
            goal='Give tips on growing plants indoors and outdoors.',
            backstory="A green-thumbed guide to all things botanical.",
            verbose=True
            )

            self.parenting_agent = Agent(
            role='Parenting Coach',
            goal='Help parents with tips for different age groups.',
            backstory="A supportive guide who helps parents thrive through each stage.",
            verbose=True
            )

            self.resume_review_agent = Agent(
            role='Resume Reviewer',
            goal='Critique and improve user resumes.',
            backstory="An expert in resume formatting, wording, and ATS strategies.",
            verbose=True
            )

            self.horoscope_agent = Agent(
            role='Horoscope Reader',
            goal='Share daily and monthly horoscope insights.',
            backstory="A cosmic expert with knowledge of zodiac signs and astrology charts.",
            verbose=True
            )

            self.event_agent = Agent(
            role='Event Planner',
            goal='Assist in planning and organizing events.',
            backstory="A logistics-savvy planner who brings celebrations to life.",
            verbose=True
            )

            self.shopping_agent = Agent(
            role='Shopping Assistant',
            goal='Help users find products and deals online.',
            backstory="A digital shopper with an eye for bargains and top-rated items.",
            verbose=True
            )

            self.mental_health_agent = Agent(
            role='Mental Health Support Agent',
            goal='Promote mindfulness and stress management.',
            backstory="A calm and caring companion helping users maintain mental balance.",
            verbose=True
            )

            self.productivity_agent = Agent(
            role='Productivity Coach',
            goal='Boost user focus and daily productivity.',
            backstory="A systems expert who streamlines work habits and time usage.",
            verbose=True
            )

            self.budget_agent = Agent(
            role='Budget Tracker',
            goal='Help users monitor and optimize expenses.',
            backstory="A financial tracking wizard who ensures every penny is accounted for.",
            verbose=True
            )

            self.affirmation_agent = Agent(
            role='Affirmation Agent',
            goal='Generate empowering affirmations for users.',
            backstory="A positivity engine that delivers confidence-boosting statements.",
            verbose=True
            )

            self.nutrition_agent = Agent(
            role='Nutritionist',
            goal='Provide healthy and balanced dietary advice.',
            backstory="A food science expert with personalized nutritional knowledge.",
            verbose=True
            )

            self.pet_agent = Agent(
            role='Pet Care Advisor',
            goal='Offer pet health and training advice.',
            backstory="A friendly vet assistant who supports pet parents daily.",
            verbose=True
            )

            self.decor_agent = Agent(
            role='Home Decor Advisor',
            goal='Suggest interior design ideas for any space.',
            backstory="An aesthetic-focused designer with an eye for cozy, modern interiors.",
            verbose=True
            )

            self.sleep_agent = Agent(
            role='Sleep Coach',
            goal='Improve users’ sleep hygiene and schedules.',
            backstory="A certified sleep scientist helping people wake up refreshed.",
            verbose=True
            )

            self.tech_agent = Agent(
            role='Tech Support Agent',
            goal='Troubleshoot common tech problems.',
            backstory="A digital assistant well-versed in gadgets, software, and devices.",
            verbose=True
            )

            self.time_agent = Agent(
            role='Time Management Agent',
            goal='Help users schedule and prioritize tasks.',
            backstory="A planner bot that helps you make the most of your hours.",
            verbose=True
            )

            self.career_agent = Agent(
            role='Career Counselor',
            goal='Advise users on career switches or growth.',
            backstory="A wise coach who helps users find fulfilling professional paths.",
            verbose=True
            )

            self.academic_agent = Agent(
            role='Academic Tutor',
            goal='Explain academic concepts like math or science.',
            backstory="A scholarly mentor guiding learners through tough subjects.",
            verbose=True
            )

            self.festival_agent = Agent(
            role='Festival Finder',
            goal='Discover local or global cultural events.',
            backstory="An explorer who knows where every festival is happening.",
            verbose=True
            )

            self.resume_keywords_agent = Agent(
            role='Resume Keyword Optimizer',
            goal='Improve resume visibility with ATS keywords.',
            backstory="A recruiter-savvy bot that boosts your resume's search ranking.",
            verbose=True
            )

            self.cover_letter_agent = Agent(
            role='Cover Letter Builder',
            goal='Draft personalized and impactful cover letters.',
            backstory="A persuasive writer who crafts compelling professional intros.",
            verbose=True
            )

            self.voice_agent = Agent(
            role='Voice Tone Coach',
            goal='Improve speaking confidence and tone.',
            backstory="A speech trainer helping users sound articulate and assured.",
            verbose=True
            )

            self.crypto_agent = Agent(
            role='Crypto Advisor',
            goal='Offer basic insights into cryptocurrency trends.',
            backstory="A blockchain enthusiast simplifying tokens and coins.",
            verbose=True
            )

            self.loan_agent = Agent(
            role='Loan & EMI Calculator',
            goal='Help calculate loan repayment options.',
            backstory="A finance bot that breaks down complex loan math.",
            verbose=True
            )

            self.insurance_agent = Agent(
            role='Insurance Helper',
            goal='Explain insurance concepts and advice.',
            backstory="An advisor who simplifies jargon and helps compare plans.",
            verbose=True
            )

            self.car_agent = Agent(
            role='Car Advisor',
            goal='Help users choose or sell cars.',
            backstory="An automobile expert helping with purchases, pricing, and features.",
            verbose=True
            )

            self.appliance_agent = Agent(
            role='Appliance Recommender',
            goal='Suggest home or kitchen gadgets.',
            backstory="A practical reviewer who compares popular appliances.",
            verbose=True
            )

            self.minimalism_agent = Agent(
            role='Minimalism Coach',
            goal='Teach users to declutter and simplify life.',
            backstory="A lifestyle guru focused on peace, clarity, and less stuff.",
            verbose=True
            )

            self.coding_prep_agent = Agent(
            role='Coding Interview Prep',
            goal='Prepare users for technical interviews.',
            backstory="A CS coach with a vault of coding problems and guidance.",
            verbose=True
            )

            self.habit_agent = Agent(
            role='Habit Tracker Agent',
            goal='Help users build good daily habits.',
            backstory="A behavioral assistant for consistent self-improvement.",
            verbose=True
            )

            self.fashion_agent = Agent(
            role='Fashion Stylist',
            goal='Recommend stylish outfits.',
            backstory="A wardrobe assistant who knows seasonal trends and personal styles.",
            verbose=True
            )

            self.music_agent = Agent(
            role='Music Recommender',
            goal='Suggest music by mood, genre, or artist.',
            backstory="A playlist curator with great taste for any vibe.",
            verbose=True
            )


            
        except Exception as e:
            logging.error(f"Error initializing agents: {str(e)}")
            # Fall back to simpler initialization if crewai has issues
            self._init_fallback()
    
    def test_agent(self, agent_name, prompt):
        """Test a specific agent with direct processing functions."""
        logging.info(f"Testing agent: {agent_name} with prompt: {prompt}")

        # Map agent names to their corresponding direct processing functions
        direct_processors = {
            'weather': lambda p: get_weather(p),
            'news': lambda p: get_news(p),
            'math': lambda p: calculate_expression(p),
            'code': lambda p: generate_code(p),
            'dictionary': lambda p: get_dictionary_word(p),
            'bmi': lambda p: calculate_bmi(*map(float, p.split(','))),
            'movie': lambda p: process_movie_booking(p),
            'restaurant': lambda p: process_restaurant_booking(p),
            'translation': lambda p: process_translation(p, 'auto', 'en'),
            'plagiarism': lambda p: process_plagiarism_detection(p),
            'gcp_vm': lambda p: process_gcp_vm_creation('tokyo-mark-452209-h9', 'us-central1-a', p, 'e2-medium', 'projects/debian-cloud/global/images/family/debian-11'),
            'gcp_autoscale': lambda p: process_gcp_autoscaling('tokyo-mark-452209-h9', 'us-central1-a', f"{p}-template", 'test-vm', p, f"{p}-base"),
            'youtube': lambda p: process_youtube_summary('@default', p),
            'stock': lambda p: process_stock_info(p),
            'recipe': lambda p: process_recipe_request(p),
            'idea': lambda p: process_idea_generation(p),
            'meditation': lambda p: process_meditation_prompt(p),
            'book': lambda p: process_book_recommendation(p),
            'convert': lambda p: process_unit_conversion(p),
            'quote': lambda p: process_quote(p),
            'movie_recommend': lambda p: process_movie_recommendation(p),
            'skincare': lambda p: process_skincare_advice(p),
            'haircare': lambda p: process_haircare_advice(p),
            'mythology': lambda p: process_mythology_info("general", p),
            'fitness': lambda p: process_fitness_coach(p),
            'meal': lambda p: process_meal_planner(p),
            'finance': lambda p: process_financial_advice(p),
            'travel': lambda p: process_travel_guide(p),
            'language': lambda p: process_language_tutor(p),
            'dream': lambda p: process_dream_interpretation(p),
            'relationship': lambda p: process_relationship_advice(p),
            'job': lambda p: process_job_finder(p),
            'gardening': lambda p: process_gardening_advice(p),
            'parenting': lambda p: process_parenting_tips(p),
            'resume_review': lambda p: process_resume_review(p),
            'horoscope': lambda p: process_horoscope(p),
            'event': lambda p: process_event_planner(p),
            'shopping': lambda p: process_shopping(p),
            'mental_health': lambda p: process_mental_health(p),
            'productivity': lambda p: process_productivity(p),
            'budget': lambda p: process_budget_tracker(p),
            'affirmation': lambda p: process_affirmations(p),
            'nutrition': lambda p: process_nutrition(p),
            'pet': lambda p: process_pet_care(p),
            'decor': lambda p: process_home_decor(p),
            'sleep': lambda p: process_sleep(p),
            'tech': lambda p: process_tech_support(p),
            'time': lambda p: process_time_management(p),
            'career': lambda p: process_career_counselor(p),
            'academic': lambda p: process_academic_tutor(p),
            'festival': lambda p: process_festival_finder(p),
            'resume_keywords': lambda p: process_resume_keywords(p),
            'cover_letter': lambda p: process_cover_letter(p),
            'voice': lambda p: process_voice_tone(p),
            'crypto': lambda p: process_crypto(p),
            'loan': lambda p: process_loan_calculator(p),
            'insurance': lambda p: process_insurance(p),
            'car': lambda p: process_car_advice(p),
            'appliance': lambda p: process_appliance(p),
            'minimalism': lambda p: process_minimalism(p),
            'coding_prep': lambda p: process_coding_interview(p),
            'habit': lambda p: process_habits(p),
            'fashion': lambda p: process_fashion(p),
            'music': lambda p: process_music_recommender(p),



        }

        # For agents with direct processing functions, use them
        if agent_name in direct_processors:
            return direct_processors[agent_name](prompt)

        # For other agents that use the OpenAI API
        if agent_name in ['content', 'resume', 'interview', 'history']:
            agent_prompts = {
                'content': f"You are a Content Writer. Write an article on {prompt}.",
                'resume': f"You are a Resume Builder. Create an ATS-optimized resume for a {prompt} role.",
                'interview': f"You are an Interview Coach. Prepare interview questions for {prompt}.",
                'history': f"You are a History Expert. Provide a historical analysis of {prompt}."
            }
            return get_response(agent_prompts[agent_name])

        return f"Unknown agent: {agent_name}"
            
    def _init_fallback(self):
        """Fallback initialization in case there are issues with crewai"""
        logging.warning("Using fallback initialization for agents")
        # Define placeholder attributes to avoid errors
        self.search_tool = None
        self.content_writer = None
        self.resume_builder = None
        self.interview_prep = None
        self.history_agent = None
        self.weather_agent = None
        self.news_agent = None
        self.math_agent = None
        self.code_agent = None
        self.dictionary_agent = None
        self.bmi_agent = None
        self.movie_agent = None
        self.restaurant_agent = None
        self.translation_agent = None
        self.plagiarism_detector = None
        self.gcp_vm_creator = None
        self.gcp_autoscaling_manager = None
        self.youtube_summarizer = None

    def create_task(self, task_type: str, **kwargs) -> Task:
        """Dynamically creates a task based on the type and input arguments."""
        task_descriptions = {
            # File 1 Tasks
            'content': f"Write an article on {kwargs.get('topic', 'a general topic')} in {kwargs.get('style', 'default')} style (~{kwargs.get('length', 500)} words).",
            'resume': f"Create an ATS-optimized resume for a {kwargs.get('role', 'unspecified')} role.",
            'interview': f"Prepare interview questions for {kwargs.get('job_role', 'a job')} ({kwargs.get('experience_level', 'any level')}).",
            'history': f"Provide a historical analysis of {kwargs.get('topic', 'a historical event')} ({kwargs.get('period', 'unknown period')}).",
        
            # Utility Tasks
            'weather': f"Fetch weather information for {kwargs.get('city', 'New York')}.",
            'news': f"Get the latest news headlines{' for category ' + kwargs.get('category') if kwargs.get('category') else ''}.",
            'math': f"Calculate the expression: {kwargs.get('expression', '1+1')}.",
            'code': f"Generate code for: {kwargs.get('prompt', 'a simple task')}.",
            'dictionary': f"Provide definition for the word: {kwargs.get('word', 'example')}.",
            'bmi': f"Calculate BMI for weight {kwargs.get('weight', 70)} kg and height {kwargs.get('height', 1.75)} m.",
        
            # New specialized tasks
            'movie': f"Book movie tickets: {kwargs.get('prompt', 'Find and book movie tickets')}",
            'restaurant': f"Book restaurant reservation: {kwargs.get('prompt', 'Find and book a restaurant table')}",
        
             # New integrated tasks
            'translation': f"Translate text from {kwargs.get('src_lang', 'auto')} to {kwargs.get('dest_lang', 'en')}: {kwargs.get('text', 'Hello world')}",
            'plagiarism': f"Check for plagiarism in the following text: {kwargs.get('text', 'Sample text to check')}",
            'gcp_vm': f"Create a VM in project {kwargs.get('project_id', 'tokyo-mark-452209-h9')} with name {kwargs.get('instance_name', 'test-vm')}",
            'gcp_autoscale': f"Set up autoscaling for VM group: {kwargs.get('group_name', 'test-group')}",
            'youtube': f"Summarize videos from channel {kwargs.get('channel_handle', '@default')} about {kwargs.get('topic', 'AI')}",
            'recipe': f"Find a recipe for: {kwargs.get('query', 'pasta')}",
            'stock': f"Fetch stock information for ticker: {kwargs.get('ticker', 'AAPL')}",
            'idea': f"Generate ideas about: {kwargs.get('prompt', 'anything creative')}",       
            'meditation': f"Guide a 5-minute meditation based on: {kwargs.get('prompt', 'relaxation')}",
            'book': f"Recommend books about: {kwargs.get('topic', 'personal growth')}",
            'convert': f"Convert units: {kwargs.get('expression', '5 miles to km')}",
            'quote': f"Find a quote about: {kwargs.get('topic', 'success')}",
            'movie_recommend': f"Recommend movies like: {kwargs.get('criteria', 'John Wick')}",
            'skincare': f"Skincare tips for: {kwargs.get('query', 'oily skin')}",
            'haircare': f"Haircare advice for: {kwargs.get('query', 'frizzy hair')}",
            'mythology': f"Explain {kwargs.get('query', 'Ra')} from {kwargs.get('civilization', 'Egyptian')} mythology.",
            'fitness': f"Create a custom workout plan based on: {kwargs.get('prompt', 'general fitness')}",
            'meal': f"Suggest a meal plan for: {kwargs.get('prompt', 'a balanced weekly diet')}",
            'finance': f"Provide personal finance tips for: {kwargs.get('prompt', 'saving and budgeting')}",
            'travel': f"Recommend travel destinations based on: {kwargs.get('prompt', 'a vacation idea')}",
            'language': f"Help learn a new language. Focus: {kwargs.get('prompt', 'basic conversation in Spanish')}",
            'dream': f"Interpret this dream: {kwargs.get('prompt', 'being lost in a forest')}",
            'relationship': f"Give relationship advice about: {kwargs.get('prompt', 'communication issues')}",
            'job': f"Suggest suitable jobs or career paths for: {kwargs.get('prompt', 'creative thinkers')}",
            'gardening': f"Give gardening tips for: {kwargs.get('prompt', 'indoor herbs')}",
            'parenting': f"Provide parenting advice for: {kwargs.get('prompt', 'toddlers')}",
            'resume_review': f"Review this resume and suggest improvements: {kwargs.get('prompt', 'text of resume')}",
            'horoscope': f"Give horoscope insight for: {kwargs.get('prompt', 'Taurus today')}",
            'event': f"Help plan an event for: {kwargs.get('prompt', 'a birthday party')}",
            'shopping': f"Help find the best deals for: {kwargs.get('prompt', 'laptop under $1000')}",
            'mental_health': f"Give mental health support and mindfulness tips for: {kwargs.get('prompt', 'stress')}",
            'productivity': f"Improve productivity for: {kwargs.get('prompt', 'working from home')}",
            'budget': f"Help track and optimize budget for: {kwargs.get('prompt', 'monthly groceries')}",
            'affirmation': f"Generate affirmations about: {kwargs.get('prompt', 'confidence')}",
            'nutrition': f"Suggest a nutritional plan for: {kwargs.get('prompt', 'weight gain')}",
            'pet': f"Give pet care advice for: {kwargs.get('prompt', 'new puppy training')}",
            'decor': f"Suggest home decor ideas for: {kwargs.get('prompt', 'small apartment')}",
            'sleep': f"Help improve sleep quality for: {kwargs.get('prompt', 'insomnia')}",
            'tech': f"Troubleshoot the following tech issue: {kwargs.get('prompt', 'Wi-Fi not connecting')}",
            'time': f"Help manage time for: {kwargs.get('prompt', 'college student with classes')}",
            'career': f"Give career guidance for: {kwargs.get('prompt', 'marketing professional switching careers')}",
            'academic': f"Explain the concept of: {kwargs.get('prompt', 'photosynthesis')}",
            'festival': f"Find upcoming festivals for: {kwargs.get('prompt', 'India in October')}",
            'resume_keywords': f"Optimize this resume with ATS keywords: {kwargs.get('prompt', 'resume text')}",
            'cover_letter': f"Draft a cover letter for: {kwargs.get('prompt', 'data analyst role')}",
            'voice': f"Help improve speaking tone for: {kwargs.get('prompt', 'public presentation')}",
            'crypto': f"Explain cryptocurrency basics about: {kwargs.get('prompt', 'Bitcoin vs Ethereum')}",
            'loan': f"Calculate loan and EMI details for: {kwargs.get('prompt', 'home loan $200k over 20 years')}",
            'insurance': f"Help understand insurance policy for: {kwargs.get('prompt', 'health insurance plan')}",
            'car': f"Suggest best car options for: {kwargs.get('prompt', 'budget of $20,000')}",
            'appliance': f"Recommend appliances for: {kwargs.get('prompt', 'energy-efficient refrigerator')}",
            'minimalism': f"Give tips to declutter and simplify life for: {kwargs.get('prompt', 'living room')}",
            'coding_prep': f"Practice coding interview questions on: {kwargs.get('prompt', 'binary trees')}",
            'habit': f"Help build good habits around: {kwargs.get('prompt', 'morning routine')}",
            'fashion': f"Suggest outfit ideas for: {kwargs.get('prompt', 'a formal dinner')}",
            'music': f"Recommend songs based on: {kwargs.get('prompt', 'chill evening')}",


        }

        agents = {
            # File 1 Agents
            'content': self.content_writer,
            'resume': self.resume_builder,
            'interview': self.interview_prep,
            'history': self.history_agent,
        
            # Utility Agents
            'weather': self.weather_agent,
            'news': self.news_agent,
            'math': self.math_agent,
            'code': self.code_agent,
            'dictionary': self.dictionary_agent,
            'bmi': self.bmi_agent,
        
            # New specialized agents
            'movie': self.movie_agent,
            'restaurant': self.restaurant_agent,
        
            # New integrated agents
            'translation': self.translation_agent,
            'plagiarism': self.plagiarism_detector,
            'gcp_vm': self.gcp_vm_creator,
            'gcp_autoscale': self.gcp_autoscaling_manager,
            'youtube': self.youtube_summarizer,
            'recipe': self.recipe_agent,
            'stock': self.stock_agent,
            'idea': self.idea_agent,
            'meditation': self.meditation_agent,
            'book': self.book_agent,
            'convert': self.unit_converter_agent,
            'quote': self.quote_agent,
            'movie_recommend': self.movie_recommender_agent,
            'skincare': self.skincare_agent,
            'haircare': self.haircare_agent,
            'mythology': self.mythology_agent,
            'fitness': self.fitness_agent,
            'meal': self.meal_agent,
            'finance': self.finance_agent,
            'travel': self.travel_agent,
            'language': self.language_agent,
            'dream': self.dream_agent,
            'relationship': self.relationship_agent,
            'job': self.job_agent,
            'gardening': self.gardening_agent,
            'parenting': self.parenting_agent,
            'resume_review': self.resume_review_agent,
            'horoscope': self.horoscope_agent,
            'event': self.event_agent,
            'shopping': self.shopping_agent,
            'mental_health': self.mental_health_agent,
            'productivity': self.productivity_agent,
            'budget': self.budget_agent,
            'affirmation': self.affirmation_agent,
            'nutrition': self.nutrition_agent,
            'pet': self.pet_agent,
            'decor': self.decor_agent,
            'sleep': self.sleep_agent,
            'tech': self.tech_agent,
            'time': self.time_agent,
            'career': self.career_agent,
            'academic': self.academic_agent,
            'festival': self.festival_agent,
            'resume_keywords': self.resume_keywords_agent,
            'cover_letter': self.cover_letter_agent,
            'voice': self.voice_agent,
            'crypto': self.crypto_agent,
            'loan': self.loan_agent,
            'insurance': self.insurance_agent,
            'car': self.car_agent,
            'appliance': self.appliance_agent,
            'minimalism': self.minimalism_agent,    
            'coding_prep': self.coding_prep_agent,
            'habit': self.habit_agent,
            'fashion': self.fashion_agent,
            'music': self.music_agent,


        }

        if task_type not in task_descriptions:
            raise ValueError(f"Unknown task type: {task_type}")

        return Task(
            description=task_descriptions[task_type],
            agent=agents[task_type],
            expected_output="A refined and high-quality output."
        )

    def execute_task(self, task_type: str, **kwargs) -> str:
        """Executes the given task in parallel using multiple agents with enhanced feedback."""
        logging.info(f"Executing task: {task_type} with parameters: {kwargs}")

        # Special handling for utility tasks that have direct functions
        if task_type == 'weather':
            return get_weather(kwargs.get('city', 'New York'))
        elif task_type == 'news':
            return get_news(kwargs.get('category', None))
        elif task_type == 'math':
            return calculate_expression(kwargs.get('expression', '1+1'))
        elif task_type == 'code':
            return generate_code(kwargs.get('prompt', 'a simple task'))
        elif task_type == 'dictionary':
            return get_dictionary_word(kwargs.get('word', 'example'))
        elif task_type == 'bmi':
            return calculate_bmi(kwargs.get('weight', 70), kwargs.get('height', 1.75))
        
        # Special handling for movie and restaurant booking
        elif task_type == 'movie':
            return process_movie_booking(kwargs.get('prompt', ''))
        elif task_type == 'restaurant':
            return process_restaurant_booking(kwargs.get('prompt', ''))
        
        # Special handling for the new integrated agents
        elif task_type == 'translation':
            return process_translation(
                kwargs.get('text', 'Hello world'),
                kwargs.get('src_lang', 'auto'),
                kwargs.get('dest_lang', 'en')
            )
        elif task_type == 'plagiarism':
            return process_plagiarism_detection(
                kwargs.get('text', 'Sample text to check'),
                kwargs.get('threshold', 0.5),
                kwargs.get('num_results', 5)
            )
        elif task_type == 'gcp_vm':
            # Handle missing or None parameters with defaults
            project_id = kwargs.get('project_id') 
            zone = kwargs.get('zone')
            instance_name = kwargs.get('instance_name', 'test-vm')
            machine_type = kwargs.get('machine_type')
            source_image = kwargs.get('source_image')
        
            return process_gcp_vm_creation(
            project_id=project_id,
            zone=zone,
            instance_name=instance_name,
            machine_type=machine_type,
            source_image=source_image
            )
        elif task_type == 'gcp_autoscale':

            project_id = kwargs.get('project_id')
            zone = kwargs.get('zone')
            template_name = kwargs.get('template_name')
            source_vm_name = kwargs.get('source_vm_name')
            group_name = kwargs.get('group_name')
            base_instance_name = kwargs.get('base_instance_name')
            return process_gcp_autoscaling(
                kwargs.get('project_id', 'tokyo-mark-452209-h9'),
                kwargs.get('zone', 'us-central1-a'),
                kwargs.get('template_name', 'default-template'),
                kwargs.get('source_vm_name', 'default-vm'),
                kwargs.get('group_name', 'default-group'),
                kwargs.get('base_instance_name', 'default-instance')
            )
        elif task_type == 'youtube':
            return process_youtube_summary(
                kwargs.get('channel_handle', '@default'),
                kwargs.get('topic', 'AI')
            )
        
        try:
            # For other tasks, use the Agent/Task/Crew structure
            task = self.create_task(task_type, **kwargs)

            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential  # ✅ Sequential Execution for web API
            )

            # ✅ For web interface, use direct OpenAI response to avoid potential issues
            prompt = task.description
            response = get_response(prompt)

            logging.info(f"Task completed. Output length: {len(response)}")
            return response
        except Exception as e:
            logging.error(f"Error executing task with agents: {str(e)}")
            # Fallback to direct OpenAI call if there are issues with crewai
            prompt = self._get_agent_prompt(task_type, **kwargs)
            response = get_response(prompt)
            return response

    def _get_agent_prompt(self, task_type: str, **kwargs) -> str:
        """Create a prompt for direct OpenAI call as fallback"""
        prompts = {
            'content': f"You are a Content Writer. Write an article on {kwargs.get('topic', 'a general topic')} in {kwargs.get('style', 'default')} style (~{kwargs.get('length', 500)} words).",
            'resume': f"You are a Resume Builder. Create an ATS-optimized resume for a {kwargs.get('role', 'unspecified')} role.",
            'interview': f"You are an Interview Coach. Prepare interview questions for {kwargs.get('job_role', 'a job')} ({kwargs.get('experience_level', 'any level')}).",
            'history': f"You are a History Expert. Provide a historical analysis of {kwargs.get('topic', 'a historical event')} ({kwargs.get('period', 'unknown period')}).",
            'movie': f"You are a Movie Booking Specialist. Help with booking movie tickets for: {kwargs.get('prompt', '')}",
            'restaurant': f"You are a Restaurant Booking Specialist. Help with making a restaurant reservation for: {kwargs.get('prompt', '')}",
            'translation': f"You are a Translator. Translate the following text from {kwargs.get('src_lang', 'auto')} to {kwargs.get('dest_lang', 'en')}: {kwargs.get('text', 'Hello world')}",
            'plagiarism': f"You are a Plagiarism Detector. Check the following text for potential plagiarism and highlight any concerns: {kwargs.get('text', 'Sample text to check')}",
            'gcp_vm': f"You are a GCP VM Specialist. Explain how to create a VM with these specifications: project={kwargs.get('project_id', 'tokyo-mark-452209-h9')}, name={kwargs.get('instance_name', 'test-vm')}",
            'gcp_autoscale': f"You are a GCP Autoscaling Specialist. Explain how to set up autoscaling for this VM group: {kwargs.get('group_name', 'test-group')}",
            'youtube': f"You are a Video Content Summarizer. Summarize videos about {kwargs.get('topic', 'AI')} from the channel {kwargs.get('channel_handle', '@default')}",
            'recipe': f"You are a recipe agent. Suggest a recipe based on this request: {kwargs.get('query', 'chicken dinner')}",
            'stock': f"You are a stock market analyst. Provide current stock data and insights for: {kwargs.get('ticker', 'AAPL')}",
            'idea': f"You are a creative idea generator. Generate ideas for: {kwargs.get('prompt', 'business growth')}",
            'meditation': f"You are a meditation coach. Provide a guided meditation focused on: {kwargs.get('prompt', 'calm and clarity')}",
            'book': f"You are a book recommender. Suggest top books about: {kwargs.get('topic', 'self improvement')}",
            'convert': f"You are a unit converter. Convert: {kwargs.get('expression', '5 miles to kilometers')}",
            'quote': f"You are a quote collector. Give an inspiring quote about: {kwargs.get('topic', 'courage')}",
            'movie_recommend': f"You are a movie expert. Recommend movies based on: {kwargs.get('criteria', 'action thrillers like John Wick')}",
            'skincare': f"You are a skincare expert. Provide advice for: {kwargs.get('query', 'dry and sensitive skin')}",
            'haircare': f"You are a haircare expert. Suggest routines or tips for: {kwargs.get('query', 'curly frizzy hair')}",
            'mythology': f"You are a mythology expert. Share stories from {kwargs.get('civilization', 'Greek')} mythology about: {kwargs.get('query', 'Zeus')}",

        }
        return prompts.get(task_type, f"You are an AI assistant. Help with this task: {kwargs}")

    def analyze_user_prompt(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze user prompt to determine which agent to use and what parameters to pass
        Returns tuple of (task_type, kwargs)
        """
        # Use GPT-3.5 to classify the user prompt
        prompt_for_gpt = f"""
        Analyze the following user prompt and determine which agent should handle it:
        "{user_prompt}"

        Choose from these agent types and extract relevant parameters:
        1. content - Extract: topic, style (optional), length (optional)
        2. resume - Extract: role, job_description (optional)
        3. interview - Extract: job_role, experience_level (optional)
        4. history - Extract: topic, period (optional)
        5. weather - Extract: city
        6. news - Extract: category (optional)
        7. math - Extract: expression (mathematical expression to evaluate)
        8. code - Extract: prompt (coding task description)
        9. dictionary - Extract: word
        10. bmi - Extract: weight (in kg), height (in meters)
        11. movie - Extract: prompt (movie booking details, including movie title, theater, date, etc.)
        12. restaurant - Extract: prompt (restaurant booking details, including cuisine, location, date, etc.)
        13. translation - Extract: text, src_lang (optional), dest_lang
        14. plagiarism - Extract: text
        15. gcp_vm - Extract: project_id (optional), zone (optional), instance_name, machine_type (optional)
        16. gcp_autoscale - Extract: template_name, group_name, source_vm_name
        17. youtube - Extract: channel_handle, topic
        18. recipe - Extract: query (ingredients or type of meal)
        19. stock - Extract: ticker
        20. idea - Extract: prompt
        21. meditation - Extract: prompt
        22. book - Extract: topic
        23. convert - Extract: expression
        24. quote - Extract: topic
        25. movie_recommend - Extract: criteria
        26. skincare - Extract: query
        27. haircare - Extract: query
        28. mythology - Extract: civilization, query
        29. fitness - Extract: prompt
        30. meal - Extract: prompt
        31. finance - Extract: prompt
        32. travel - Extract: prompt
        33. language - Extract: prompt
        34. dream - Extract: prompt
        35. relationship - Extract: prompt
        36. job - Extract: prompt
        37. gardening - Extract: prompt
        38. parenting - Extract: prompt
        39. resume_review - Extract: prompt
        40. horoscope - Extract: prompt
        41. event - Extract: prompt
        42. shopping - Extract: prompt
        43. mental_health - Extract: prompt
        44. productivity - Extract: prompt
        45. budget - Extract: prompt
        46. affirmation - Extract: prompt
        47. nutrition - Extract: prompt
        48. pet - Extract: prompt
        49. decor - Extract: prompt
        50. sleep - Extract: prompt
        51. tech - Extract: prompt
        52. time - Extract: prompt
        53. career - Extract: prompt
        54. academic - Extract: prompt
        55. festival - Extract: prompt
        56. resume_keywords - Extract: prompt
        57. cover_letter - Extract: prompt
        58. voice - Extract: prompt
        59. crypto - Extract: prompt
        60. loan - Extract: prompt
        61. insurance - Extract: prompt
        62. car - Extract: prompt
        63. appliance - Extract: prompt
        64. minimalism - Extract: prompt
        65. coding_prep - Extract: prompt
        66. habit - Extract: prompt
        67. fashion - Extract: prompt
        68. music - Extract: prompt




        Return your analysis as a JSON object with this structure:
        {{
            "agent_type": "chosen_agent_type",
            "params": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}

        Make sure to return valid JSON that can be parsed.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a task classification assistant. Respond with valid JSON only."},
                {"role": "user", "content": prompt_for_gpt}
            ],
            max_tokens=512,
            temperature=0.3
        )

        try:
            # Extract the text content and parse it as JSON
            response_text = response.choices[0].message.content.strip()
            # Find the JSON object in the response (handling potential extra text)
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                classification = json.loads(json_str)
            else:
                classification = json.loads(response_text)


            if "params" in classification:
                     # List of placeholder values to remove
                    placeholders = ["optional", "not specified", "not provided", "unknown"]
                    params = classification["params"]
                    # Create a copy to avoid modifying during iteration
                    params_to_remove = []
            
                    for key, value in params.items():
                         # If the value is a string and matches any placeholder
                         if isinstance(value, str) and value.lower() in placeholders:
                                params_to_remove.append(key)
            
                    # Remove identified placeholders
                    for key in params_to_remove:
                         params.pop(key)
        
            
            return classification["agent_type"], classification["params"]
        except Exception as e:
            logging.error(f"Error parsing classification response: {e}")
            logging.error(f"Response was: {response.choices[0].message.content}")
            # Default to content agent if classification fails
            return "content", {"topic": user_prompt}

# ✅ Main function to handle user input (CLI mode)
def main():
    agent_system = AgentSystem()
    print("=" * 60)
    print("🤖 Multi-Agent System 🤖")
    print("Type 'exit' to quit the program")
    print("Type 'test <agent_name> <prompt>' to test a specific agent")
    print("=" * 60)
    print("Available agents:")
    print("- Content Writer: Write articles on any topic")
    print("- Resume Builder: Create professional resumes")
    print("- Interview Coach: Prepare for job interviews")
    print("- History Expert: Get historical analysis")
    print("- Weather Expert: Get weather for a city")
    print("- News Reporter: Get latest news headlines")
    print("- Math Wizard: Calculate mathematical expressions")
    print("- Code Generator: Generate code based on requirements")
    print("- Dictionary: Get definitions of words")
    print("- BMI Calculator: Calculate your BMI")
    print("- Movie Booking: Book movie tickets")
    print("- Restaurant Booking: Make restaurant reservations")
    print("- Translation: Translate text between languages")
    print("- Plagiarism Detector: Check text for potential plagiarism")
    print("- GCP VM Creator: Create virtual machines on Google Cloud")
    print("- GCP Autoscaling: Set up autoscaling for VM groups")
    print("- YouTube Summarizer: Summarize YouTube videos on a topic")
    print("- Recipe Agent: Provides Recipe")
    print("- StocK Agent: Provides stock information")
    print("=" * 60)
    print("Example prompts:")
    print("- Write an article about artificial intelligence")
    print("- Create a resume for a software engineer")
    print("- What's the weather in London?")
    print("- Calculate 5 * (3 + 2)")
    print("- Define the word 'serendipity'")
    print("- Book two tickets for the new superhero movie on Friday")
    print("- Make a reservation for dinner at an Italian restaurant tomorrow")
    print("- Translate 'Hello, how are you?' to Spanish")
    print("- Check if this text is plagiarized: 'AI is transforming the world'")
    print("- Summarize videos about machine learning from @techworld")
    print("=" * 60)
    print("Test mode examples:")
    print("- test weather London")
    print("- test translation Hello, how are you?")
    print("- test youtube machine learning basics")
    print("=" * 60)
    print("Find a quick dinner recipe using chicken and broccoli")
    print("Give me the stock value for Microsoft")
    print("- Idea Generator: Get creative ideas")
    print("- Meditation Coach: Guided meditation sessions")
    print("- Book Recommender: Book suggestions based on interest")
    print("- Unit Converter: Convert units like km to miles")
    print("- Quote Collector: Inspirational and philosophical quotes")
    print("- Movie Recommender: Suggest films by genre or mood")
    print("- Skincare Advisor: Skincare routines for your skin type")
    print("- Haircare Advisor: Tips for healthy hair")
    print("- Mythology Expert: Learn about Egyptian, Indian, or Greek gods")
    print("=" * 60)
    print("- Fitness Coach: Custom workout plans and tips")
    print("- Meal Planner: Weekly meal planning ideas")
    print("- Financial Advisor: Basic personal finance tips")
    print("- Travel Guide: Recommend destinations and travel tips")
    print("- Language Tutor: Help learning new languages")
    print("- Dream Interpreter: Explain what your dreams might mean")
    print("- Relationship Coach: Advice on relationships and communication")
    print("- Job Finder: Suggest career options or job roles")
    print("- Gardening Expert: Tips for indoor or outdoor gardening")
    print("- Parenting Coach: Tips for managing kids of different ages")
    print("- Resume Reviewer: Critique and improve your resume")
    print("- Horoscope Reader: Daily or monthly zodiac insights")
    print("- Event Planner: Help organizing events or parties")
    print("- Shopping Assistant: Help finding deals or products online")
    print("- Mental Health Support Agent: Mindfulness and stress tips")
    print("- Productivity Coach: Tools and tips for better focus")
    print("- Budget Tracker: Help manage and optimize expenses")
    print("- Affirmation Agent: Daily affirmations for clarity and confidence")
    print("- Nutritionist: Offer balanced diet suggestions")
    print("- Pet Care Advisor: Tips for pet health and behavior")
    print("- Home Decor Advisor: Interior design suggestions")
    print("- Sleep Coach: Improve sleep quality and routine")
    print("- Tech Support Agent: Troubleshoot basic tech issues")
    print("- Time Management Agent: Help schedule and prioritize tasks")
    print("- Career Counselor: Guidance for switching or growing in careers")
    print("- Academic Tutor: Explain concepts in math, science, etc.")
    print("- Festival Finder: Discover global and local events")
    print("- Resume Keyword Optimizer: Enhance resumes with ATS keywords")
    print("- Cover Letter Builder: Draft professional cover letters")
    print("- Voice Tone Coach: Improve speaking tone and delivery")
    print("- Crypto Advisor: Explain cryptocurrency basics")
    print("- Loan & EMI Calculator: Help calculate loan repayments")
    print("- Insurance Helper: Clarify insurance coverage and terms")
    print("- Car Advisor: Suggest vehicles to buy/sell")
    print("- Appliance Recommender: Help choose gadgets for home")
    print("- Minimalism Coach: Help declutter and simplify life")
    print("- Coding Interview Prep: Practice coding interview Qs")
    print("- Habit Tracker Agent: Build and track good habits")
    print("- Fashion Stylist: Outfit suggestions based on occasion")
    print("- Music Recommender: Suggest songs by mood, genre, or artist")
    
    while True:
        user_input = input("\n🧠 What would you like help with? (type 'exit' to quit)\n> ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Thank you for using the Multi-Agent System. Goodbye!")
            break
            
        # Check if this is a test command
        if user_input.lower().startswith('test '):
            parts = user_input.split(' ', 2)
            if len(parts) >= 3:
                agent_name = parts[1].lower()
                prompt = parts[2]
                print(f"\nTesting agent: {agent_name}")
                print(f"Prompt: {prompt}")
                print("-" * 50)
                try:
                    result = agent_system.test_agent(agent_name, prompt)
                    print("\n" + "=" * 60)
                    print(f"🤖 TEST RESULT FOR {agent_name.upper()} AGENT:")
                    print("=" * 60)
                    print(result)
                    print("=" * 60)
                except Exception as e:
                    print(f"❌ Error testing agent: {str(e)}")
                    logging.error(f"Error testing agent: {e}", exc_info=True)
            else:
                print("❌ Invalid test command. Format: test <agent_name> <prompt>")
            continue
            
        print("\nProcessing your request...\n")
        try:
            # Analyze user input to determine which agent to use
            task_type, kwargs = agent_system.analyze_user_prompt(user_input)
            # Execute the task with the appropriate agent
            result = agent_system.execute_task(task_type, **kwargs)
            print("\n" + "=" * 60)
            print(f"🤖 {task_type.upper()} AGENT RESPONSE:")
            print("=" * 60)
            print(result)
            print("=" * 60)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logging.error(f"Error processing request: {e}", exc_info=True)

# ✅ Run the main program if this file is executed directly
if __name__ == "__main__":
    main()