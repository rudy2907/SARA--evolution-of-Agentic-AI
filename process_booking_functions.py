import sys
import os
import logging
import json
import openai
from typing import Dict, Any


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required. Set it in .env file or as an environment variable.")

# Initialize OpenAI client directly
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_response(prompt: str) -> str:
    """Generate response using OpenAI GPT model."""
    try:
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
    except Exception as e:
        logger.error(f"Error generating OpenAI response: {str(e)}")
        return f"Error generating response: {str(e)}"

def extract_booking_details(user_prompt: str, booking_type: str) -> Dict[str, Any]:
    """
    Extract booking details from user prompt using OpenAI
    
    Args:
        user_prompt: The user's booking request
        booking_type: "movie" or "restaurant"
        
    Returns:
        Dictionary with booking details
    """
    try:
        if booking_type == "movie":
            prompt = f"""
            Extract movie booking details from the following text:
            "{user_prompt}"
            
            Return a JSON object with these fields:
            - movie_name: Name of the movie to watch
            - show_time: Time of the show (e.g., "7:30 PM")
            - num_tickets: Number of tickets to book
            
            If a detail is not specified, provide a reasonable default.
            """
        else:  # restaurant
            prompt = f"""
            Extract restaurant booking details from the following text:
            "{user_prompt}"
            
            Return a JSON object with these fields:
            - restaurant_name: Name of the restaurant
            - time_slot: Time of the reservation (e.g., "7:00 PM")
            - num_people: Number of people in the party
            - special_requests: Any special requests (optional)
            
            If a detail is not specified, provide a reasonable default.
            """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract booking information from text and return it as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to find a JSON object in the response
        import re
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        details = json.loads(response_text)
        return details
    
    except Exception as e:
        logger.error(f"Error extracting booking details: {str(e)}")
        # Return default details based on booking type
        if booking_type == "movie":
            return {"movie_name": None, "show_time": "7:00 PM", "num_tickets": 2}
        else:
            return {"restaurant_name": None, "time_slot": "7:00 PM", "num_people": 2, "special_requests": ""}

def process_movie_booking(user_prompt: str) -> str:
    """Process a movie booking request using the local website."""
    logging.info(f"Processing movie booking request: {user_prompt}")
    
    try:
        # Try to use website integration
        try:
            # Import the website integration module
            from website_integration import open_movie_booking_section
            
            # Extract booking details
            booking_details = extract_booking_details(user_prompt, "movie")
            movie_name = booking_details.get("movie_name")
            show_time = booking_details.get("show_time", "7:00 PM")
            num_tickets = booking_details.get("num_tickets", 2)
            
            # Open the website
            result = open_movie_booking_section()
            
            # Add booking details to the response
            booking_info = ""
            if movie_name:
                booking_info = f"\n\nRecommended options based on your request:\nMovie: {movie_name}\nTime: {show_time}\nTickets: {num_tickets}"
            
            return f"Movie Booking Agent: {result}{booking_info}"
            
        except ImportError as ie:
            logger.warning(f"Website integration not available: {ie}")
        
        # Fall back to OpenAI response if website integration fails
        prompt = f"""
        You are a helpful Movie Booking Assistant helping a user book movie tickets.
        
        The user request is: "{user_prompt}"
        
        Provide a friendly response acknowledging their request. You can't actually book tickets,
        but you can provide guidance on how they could book tickets themselves.
        
        If the user mentioned a specific movie, time, or number of tickets, reference these in your response.
        """
        
        response = get_response(prompt)
        return f"Movie Booking Agent:\n{response}"
        
    except Exception as e:
        logging.error(f"Error in movie booking: {str(e)}")
        return f"Movie Booking Agent: Error processing your request: {str(e)}"

def process_restaurant_booking(user_prompt: str) -> str:
    """Process a restaurant booking request using the local website."""
    logging.info(f"Processing restaurant booking request: {user_prompt}")
    
    try:
        # Try to use website integration
        try:
            # Import the website integration module
            from website_integration import open_restaurant_booking_section
            
            # Extract booking details
            booking_details = extract_booking_details(user_prompt, "restaurant")
            restaurant_name = booking_details.get("restaurant_name")
            time_slot = booking_details.get("time_slot", "7:00 PM")
            num_people = booking_details.get("num_people", 2)
            special_requests = booking_details.get("special_requests", "")
            
            # Open the website
            result = open_restaurant_booking_section()
            
            # Add booking details to the response
            booking_info = ""
            if restaurant_name:
                booking_info = f"\n\nRecommended options based on your request:\nRestaurant: {restaurant_name}\nTime: {time_slot}\nPeople: {num_people}"
                if special_requests:
                    booking_info += f"\nSpecial requests: {special_requests}"
            
            return f"Restaurant Booking Agent: {result}{booking_info}"
            
        except ImportError as ie:
            logger.warning(f"Website integration not available: {ie}")
        
        # Fall back to OpenAI response if website integration fails
        prompt = f"""
        You are a helpful Restaurant Booking Assistant helping a user make a restaurant reservation.
        
        The user request is: "{user_prompt}"
        
        Provide a friendly response acknowledging their request. You can't actually book a table,
        but you can provide guidance on how they could make a reservation themselves.
        
        If the user mentioned a specific restaurant, time, or party size, reference these in your response.
        """
        
        response = get_response(prompt)
        return f"Restaurant Booking Agent:\n{response}"
        
    except Exception as e:
        logging.error(f"Error in restaurant booking: {str(e)}")
        return f"Restaurant Booking Agent: Error processing your request: {str(e)}"

# Main function to determine booking type and route accordingly
def process_user_request(user_prompt: str) -> str:
    """
    Process a user booking request by determining the type and handling accordingly
    
    Args:
        user_prompt: The user's booking request
        
    Returns:
        Response from the appropriate booking function
    """
    # Determine if the user wants to book a movie or restaurant
    movie_keywords = ["movie", "film", "cinema", "theater", "watch", "showing", "screening", "ticket"]
    restaurant_keywords = ["restaurant", "dinner", "lunch", "breakfast", "eat", "dining", "food", "meal", "table", "reservation"]
    
    # Count occurrences of keywords
    movie_count = sum(1 for keyword in movie_keywords if keyword.lower() in user_prompt.lower())
    restaurant_count = sum(1 for keyword in restaurant_keywords if keyword.lower() in user_prompt.lower())
    
    # Determine booking type based on keyword counts
    if movie_count > restaurant_count:
        return process_movie_booking(user_prompt)
    else:
        return process_restaurant_booking(user_prompt)