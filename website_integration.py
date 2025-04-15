import logging
import webbrowser
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Website URL - Update this to your actual website URL
WEBSITE_URL = "file:///home/rudy/project_website/index.html"

def open_booking_website():
    """Open the booking website in the default browser."""
    try:
        webbrowser.open(WEBSITE_URL)
        logger.info(f"Website opened successfully: {WEBSITE_URL}")
        return "Booking website opened in your browser. Please complete your booking there."
    except Exception as e:
        logger.error(f"Error opening website: {str(e)}")
        return f"Error opening website: {str(e)}"

def open_movie_booking_section():
    """Open the movie booking section of the website."""
    try:
        # Open the website with the movie booking section
        movie_url = f"{WEBSITE_URL}#movie-booking"
        webbrowser.open(movie_url)
        logger.info(f"Movie booking section opened: {movie_url}")
        return "Movie booking section opened in your browser. Please complete your ticket booking there."
    except Exception as e:
        logger.error(f"Error opening movie booking section: {str(e)}")
        # Fallback to opening the main page
        return open_booking_website()
        
def open_restaurant_booking_section():
    """Open the restaurant booking section of the website."""
    try:
        # Open the website with the restaurant booking section
        restaurant_url = f"{WEBSITE_URL}#restaurant-booking"
        webbrowser.open(restaurant_url)
        logger.info(f"Restaurant booking section opened: {restaurant_url}")
        return "Restaurant booking section opened in your browser. Please complete your reservation there."
    except Exception as e:
        logger.error(f"Error opening restaurant booking section: {str(e)}")
        # Fallback to opening the main page
        return open_booking_website()