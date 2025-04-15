import logging
from website_integration import WebsiteBookingTool, open_booking_website

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the booking tool
booking_tool = None

def get_booking_tool():
    """Get or initialize the booking tool."""
    global booking_tool
    if booking_tool is None:
        try:
            booking_tool = WebsiteBookingTool()
            logger.info("Booking tool initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing booking tool: {str(e)}")
            return None
    return booking_tool

def process_movie_booking_website(user_prompt):
    """Process a movie booking request using the local website."""
    logging.info(f"Processing movie booking request via website: {user_prompt}")
    
    try:
        # Use OpenAI to extract booking details from the prompt
        from openai_helper import extract_movie_booking_details
        booking_details = extract_movie_booking_details(user_prompt)
        
        if not booking_details:
            # Fallback to just opening the website
            result = open_booking_website()
            return f"Movie Booking Agent: {result}"
        
        # Get booking tool
        tool = get_booking_tool()
        if not tool:
            result = open_booking_website()
            return f"Movie Booking Agent: Could not initialize automated booking. {result}"
        
        # Get available movies if needed
        if not booking_details.get('movie_name'):
            available_movies = tool.get_available_movies()
            movies_list = ", ".join(available_movies)
            return f"Movie Booking Agent: Please specify which movie you want to watch. Available movies: {movies_list}"
        
        # Perform the booking
        result = tool.book_movie_tickets(
            movie_name=booking_details.get('movie_name', 'No movie specified'),
            show_time=booking_details.get('show_time', '7:00 PM'),  # Default time
            num_tickets=booking_details.get('num_tickets', 1)       # Default 1 ticket
        )
        
        return f"Movie Booking Agent: {result}"
    
    except Exception as e:
        logging.error(f"Error in website movie booking: {str(e)}")
        # Fallback to opening the website manually
        result = open_booking_website()
        return f"Movie Booking Agent: Error processing your request: {str(e)}. {result}"

def process_restaurant_booking_website(user_prompt):
    """Process a restaurant booking request using the local website."""
    logging.info(f"Processing restaurant booking request via website: {user_prompt}")
    
    try:
        # Use OpenAI to extract booking details from the prompt
        from openai_helper import extract_restaurant_booking_details
        booking_details = extract_restaurant_booking_details(user_prompt)
        
        if not booking_details:
            # Fallback to just opening the website
            result = open_booking_website()
            return f"Restaurant Booking Agent: {result}"
        
        # Get booking tool
        tool = get_booking_tool()
        if not tool:
            result = open_booking_website()
            return f"Restaurant Booking Agent: Could not initialize automated booking. {result}"
        
        # Get available restaurants if needed
        if not booking_details.get('restaurant_name'):
            available_restaurants = tool.get_available_restaurants()
            restaurants_list = ", ".join(available_restaurants)
            return f"Restaurant Booking Agent: Please specify which restaurant you want to book. Available restaurants: {restaurants_list}"
        
        # Perform the booking
        result = tool.book_restaurant_table(
            restaurant_name=booking_details.get('restaurant_name', 'No restaurant specified'),
            time_slot=booking_details.get('time_slot', '7:00 PM'),        # Default time
            num_people=booking_details.get('num_people', 2),              # Default 2 people
            special_requests=booking_details.get('special_requests', '')  # Default no special requests
        )
        
        return f"Restaurant Booking Agent: {result}"
    
    except Exception as e:
        logging.error(f"Error in website restaurant booking: {str(e)}")
        # Fallback to opening the website manually
        result = open_booking_website()
        return f"Restaurant Booking Agent: Error processing your request: {str(e)}. {result}"