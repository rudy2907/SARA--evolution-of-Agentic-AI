import re
import random
import string
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

class BookingSystem:
    def __init__(self):
        # Ensure tickets directory exists
        os.makedirs('tickets', exist_ok=True)
        
        # Dictionary to store booking types and their processing functions
        self.booking_types = {
            'movie': self._process_movie_booking,
            'restaurant': self._process_restaurant_booking
        }
    
    def _generate_booking_id(self):
        """Generate a unique booking ID"""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    def is_booking_request(self, user_prompt):
        """
        Detect if the user prompt is a booking request
        Uses regex to match booking-related keywords
        """
        # Define patterns for different booking types
        booking_patterns = {
            'movie': r'\b(movie|film|cinema|ticket|tickets)\b',
            'restaurant': r'\b(restaurant|dinner|table|reservation|book)\b'
        }
        
        # Check each booking type pattern
        for booking_type, pattern in booking_patterns.items():
            if re.search(pattern, user_prompt.lower()):
                return booking_type
        
        return None
    
    def _create_ticket_pdf(self, booking_details):
        """Create a PDF ticket with booking details"""
        booking_id = booking_details.get('booking_id', 'N/A')
        booking_type = booking_details.get('type', 'Booking')
        
        # Create a unique filename
        filename = f'tickets/{booking_type.lower()}_{booking_id}_ticket.pdf'
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title = Paragraph(f"{booking_type.upper()} TICKET", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Add booking details
        for key, value in booking_details.items():
            if key not in ['path', 'type', 'has_attachment', 'attachment_type', 'attachment_data', 'attachment_filename']:
                detail = Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal'])
                story.append(detail)
                story.append(Spacer(1, 6))
        
        # Add booking ID
        booking_id_para = Paragraph(f"<b>Booking ID:</b> {booking_id}", styles['Normal'])
        story.append(booking_id_para)
        
        # Generate PDF
        doc.build(story)
        
        return filename
    
    def _process_movie_booking(self, user_prompt):
        """Process a movie booking request"""
        # More comprehensive regex for movie booking
        movie_match = re.search(r'(?:book|get)\s*(?:(\d+)\s*(?:ticket(?:s)?)\s*(?:for)?)\s*(.+?)\s*movie', user_prompt.lower())
        date_match = re.search(r'(tonight|today|tomorrow|on\s+\d{1,2}\s+\w+\s+\d{4})', user_prompt.lower())
        time_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', user_prompt.lower())
        
        # Determine number of tickets
        num_tickets = 2  # Default to 2 tickets if not specified
        if movie_match and movie_match.group(1):
            num_tickets = int(movie_match.group(1))
        
        # Extract movie name
        movie_name = (movie_match.group(2).title() if movie_match 
                      else 'Interstellar 2')
        
        # Handle special date keywords
        if date_match:
            date_str = date_match.group(1)
            if date_str == 'tonight' or date_str == 'today':
                show_date = datetime.datetime.now().strftime('%d %B %Y')
            elif date_str == 'tomorrow':
                show_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%d %B %Y')
            else:
                show_date = date_match.group(1)
        else:
            show_date = datetime.datetime.now().strftime('%d %B %Y')
        
        # Handle time
        show_time = (time_match.group(1) if time_match 
                     else '7:30 PM')
        
        # Generate booking details
        booking_id = self._generate_booking_id()
        
        # Generate unique seats
        available_seats = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 
                           'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'H1', 'H2']
        seats = random.sample(available_seats, min(num_tickets, len(available_seats)))
        
        booking_details = {
            'booking_id': booking_id,
            'type': 'Movie',
            'movie_name': movie_name,
            'date': show_date,
            'time': show_time,
            'number_of_tickets': num_tickets,
            'seats': ', '.join(seats),
            'path': ''  # Will be set after PDF creation
        }
        
        # Create PDF ticket
        ticket_path = self._create_ticket_pdf(booking_details)
        booking_details['path'] = ticket_path
        
        return {
            'text': f"Your movie ticket{'s' if num_tickets > 1 else ''} for {movie_name} on {show_date} at {show_time} ({num_tickets} ticket{'s' if num_tickets > 1 else ''}) is confirmed. Booking ID: {booking_id}",
            'has_attachment': True,
            'attachment_type': 'application/pdf',
            'attachment_data': '',
            'attachment_filename': f'movie_ticket_{booking_id}.pdf'
        }
    
    def _process_restaurant_booking(self, user_prompt):
        """Process a restaurant booking request"""
        # Extract restaurant details using regex
        restaurant_match = re.search(r'at\s+(.+?)\s+restaurant', user_prompt.lower())
        date_match = re.search(r'(tonight|today|tomorrow|on\s+\d{1,2}\s+\w+\s+\d{4})', user_prompt.lower())
        time_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', user_prompt.lower())
        party_match = re.search(r'for\s+(\d+)\s+(?:people|persons)', user_prompt.lower())
        
        # Default or extracted values
        restaurant_name = (restaurant_match.group(1).title() if restaurant_match 
                           else 'Unspecified Restaurant')
        
        # Handle special date keywords
        if date_match:
            date_str = date_match.group(1)
            if date_str == 'tonight' or date_str == 'today':
                reservation_date = datetime.datetime.now().strftime('%d %B %Y')
            elif date_str == 'tomorrow':
                reservation_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%d %B %Y')
            else:
                reservation_date = date_match.group(1)
        else:
            reservation_date = datetime.datetime.now().strftime('%d %B %Y')
        
        # Handle time
        reservation_time = (time_match.group(1) if time_match 
                            else '7:30 PM')
        
        # Handle party size
        party_size = (party_match.group(1) if party_match else '2')
        
        # Generate booking details
        booking_id = self._generate_booking_id()
        booking_details = {
            'booking_id': booking_id,
            'type': 'Restaurant',
            'restaurant_name': restaurant_name,
            'date': reservation_date,
            'time': reservation_time,
            'party_size': f'{party_size} persons',
            'table_number': random.choice(['1', '5', '12', '8']),
            'path': ''  # Will be set after PDF creation
        }
        
        # Create PDF ticket
        ticket_path = self._create_ticket_pdf(booking_details)
        booking_details['path'] = ticket_path
        
        return {
            'text': f"Your table reservation at {restaurant_name} for {party_size} people on {reservation_date} at {reservation_time} is confirmed. Booking ID: {booking_id}",
            'has_attachment': True,
            'attachment_type': 'application/pdf',
            'attachment_data': '',
            'attachment_filename': f'restaurant_ticket_{booking_id}.pdf'
        }
    
    def process_booking(self, user_id, user_prompt):
        """
        Process booking based on the type of booking detected
        
        Args:
            user_id (str): Unique identifier for the user
            user_prompt (str): User's booking request
        
        Returns:
            dict: Booking confirmation details
        """
        # Detect booking type
        booking_type = self.is_booking_request(user_prompt)
        
        if not booking_type:
            return {
                'text': "Sorry, I couldn't understand your booking request. Please be more specific.",
                'has_attachment': False
            }
        
        # Call the specific booking method
        booking_method = self.booking_types.get(booking_type)
        if booking_method:
            return booking_method(user_prompt)
        
        return {
            'text': f"Booking for {booking_type} is not currently supported.",
            'has_attachment': False
        }
    
    def get_ticket(self, booking_id):
        """
        Retrieve ticket information based on booking ID
        
        Args:
            booking_id (str): Unique booking identifier
        
        Returns:
            dict: Ticket information including path
        """
        # Find ticket in the tickets directory
        for filename in os.listdir('tickets'):
            if booking_id in filename:
                return {
                    'path': os.path.join('tickets', filename),
                    'type': filename.split('_')[0].capitalize()
                }
        
        raise FileNotFoundError(f"No ticket found for booking ID: {booking_id}")