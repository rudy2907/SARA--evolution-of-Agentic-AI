from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import time
import json
import os
from agent_system_original import AgentSystem
from booking_system import BookingSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create an instance of the AgentSystem
agent_system = AgentSystem()

# Create an instance of the BookingSystem
booking_system = BookingSystem()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint to process user prompts and return agent responses"""
    data = request.json
    user_id = data.get('user_id', 'default_user')
    user_prompt = data.get('message', '')
    
    if not user_prompt:
        return jsonify({
            'status': 'error',
            'message': 'No prompt provided'
        }), 400
    
    try:
        # Check if this is a booking-related request
        if booking_system.is_booking_request(user_prompt):
            # Process with booking system
            start_time = time.time()
            response = booking_system.process_booking(user_id, user_prompt)
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'agent_type': 'booking',
                'response': response['text'],
                'processing_time': f"{processing_time:.2f}"
            }
            
            # Check if there's a ticket to attach
            if response.get('has_attachment', False):
                result['has_attachment'] = True
                result['attachment_type'] = response.get('attachment_type', 'application/pdf')
                result['attachment_data'] = response.get('attachment_data', '')
                result['attachment_filename'] = response.get('attachment_filename', 'ticket.pdf')
            
            return jsonify(result)
        else:
            # Process with regular agent system
            start_time = time.time()
            task_type, kwargs = agent_system.analyze_user_prompt(user_prompt)
            logger.info(f"Classified as {task_type} task with params: {kwargs}")
            
            # Execute the task with the appropriate agent
            result = agent_system.execute_task(task_type, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Task completed in {processing_time:.2f} seconds")
            
            # Return the result
            return jsonify({
                'status': 'success',
                'agent_type': task_type,
                'response': result,
                'processing_time': f"{processing_time:.2f}"
            })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Endpoint to return the list of available agents"""
    agents = [
        {
            "id": "content",
            "name": "Content Writer",
            "description": "Writes articles on any topic",
            "example": "Write an article about artificial intelligence"
        },
        {
            "id": "resume",
            "name": "Resume Builder",
            "description": "Creates professional resumes",
            "example": "Create a resume for a software engineer"
        },
        {
            "id": "interview",
            "name": "Interview Coach",
            "description": "Prepares for job interviews",
            "example": "Help me prepare for a product manager interview"
        },
        {
            "id": "history",
            "name": "History Expert",
            "description": "Provides historical analysis",
            "example": "Explain the causes of World War II"
        },
        {
            "id": "weather",
            "name": "Weather Expert",
            "description": "Gets weather for a city",
            "example": "What's the weather in London?"
        },
        {
            "id": "news",
            "name": "News Reporter",
            "description": "Gets latest news headlines",
            "example": "Show me the latest news"
        },
        {
            "id": "math",
            "name": "Math Wizard",
            "description": "Calculates mathematical expressions",
            "example": "Calculate 5 * (3 + 2)"
        },
        {
            "id": "code",
            "name": "Code Generator",
            "description": "Generates code based on requirements",
            "example": "Write a Python function to sort a list"
        },
        {
            "id": "dictionary",
            "name": "Dictionary",
            "description": "Gets definitions of words",
            "example": "Define the word 'serendipity'"
        },
        {
            "id": "bmi",
            "name": "BMI Calculator",
            "description": "Calculates BMI",
            "example": "Calculate BMI for weight 70 kg and height 1.75 m"
        },
        {
            "id": "booking",
            "name": "Booking Agent",
            "description": "Books movie tickets, flights, hotels, and restaurants",
            "example": "Book a flight to London"
        },
        # New agents added below
        {
            "id": "translation",
            "name": "Translator",
            "description": "Translates text between languages",
            "example": "Translate 'Hello, how are you?' to Spanish"
        },
        {
            "id": "plagiarism",
            "name": "Plagiarism Detector",
            "description": "Checks text for potential plagiarism",
            "example": "Check if this text is plagiarized: 'Artificial intelligence is transforming the world'"
        },
        {
            "id": "gcp_vm",
            "name": "GCP VM Creator",
            "description": "Creates virtual machines on Google Cloud Platform",
            "example": "Create a VM named 'test-server' in GCP"
        },
        {
            "id": "gcp_autoscale",
            "name": "GCP Autoscaling Manager",
            "description": "Sets up autoscaling policies for GCP virtual machines",
            "example": "Set up autoscaling for VM group 'web-servers'"
        },
        {
            "id": "youtube",
            "name": "YouTube Summarizer",
            "description": "Summarizes YouTube videos on specified topics",
            "example": "Summarize videos about AI from the channel @techworld"
        },

        {
             "id": "recipe",
             "name": "Recipe Recommender",
             "description": "Suggests recipes based on ingredients or meal type",
             "example": "Find a recipe for chicken and broccoli dinner"
         },
        {
             "id": "stock",
             "name": "Stock Analyst",
             "description": "Provides current stock information",
             "example": "What's the current stock price of AAPL?"
         },
         {
             "id": "idea",
             "name": "Idea Generator",
             "description": "Generates creative ideas on various topics",
             "example": "Give me 5 creative ideas for a birthday party"
        },
        {
             "id": "meditation",
             "name": "Meditation Coach",
             "description": "Provides guided meditation sessions",
             "example": "Guide me through a 5-minute relaxation meditation"
         },
         {
             "id": "book",
             "name": "Book Recommender",
             "description": "Recommends books based on interests",
             "example": "Suggest books about personal growth"
         },
         {
             "id": "convert",
             "name": "Unit Converter",
             "description": "Converts between different units of measurement",
             "example": "Convert 5 miles to kilometers"
         },
         {
             "id": "quote",
             "name": "Quote Collector",
             "description": "Provides inspirational and philosophical quotes",
             "example": "Share a quote about perseverance"
         },
         {
             "id": "movie_recommend",
             "name": "Movie Recommender",
             "description": "Suggests movies based on preferences",
             "example": "Recommend movies similar to Inception"
         },
         {
             "id": "skincare",
             "name": "Skincare Advisor",
             "description": "Provides skincare tips and routines",
             "example": "Suggest a skincare routine for oily skin"
         },
         {
             "id": "haircare",
             "name": "Haircare Advisor",
             "description": "Offers hair care tips and advice",
             "example": "How to take care of curly hair"
         },
         {
             "id": "mythology",
             "name": "Mythology Expert",
             "description": "Explains stories from ancient mythologies",
             "example": "Tell me about Zeus in Greek mythology"
         },
         {
             "id": "fitness",
             "name": "Fitness Coach",
             "description": "Provides custom workout plans and fitness tips",
             "example": "Create a home workout plan for building strength"
         },
         {
             "id": "meal",
             "name": "Meal Planner",
             "description": "Creates weekly meal plans",
             "example": "Make a balanced meal plan for a week"
         },
         {
             "id": "finance",
             "name": "Financial Advisor",
             "description": "Offers personal finance tips",
             "example": "How to create a monthly budget"
         },
         {
             "id": "travel",
             "name": "Travel Guide",
             "description": "Recommends travel destinations and planning tips",
             "example": "Suggest destinations for a beach vacation"
         },
         {
             "id": "language",
             "name": "Language Tutor",
             "description": "Helps with learning new languages",
             "example": "Teach me basic Spanish phrases for travel"
         },
         {
             "id": "dream",
             "name": "Dream Interpreter",
             "description": "Interprets the meaning of dreams",
             "example": "What does it mean to dream about falling?"
         },
         {
             "id": "relationship",
             "name": "Relationship Coach",
             "description": "Provides relationship advice",
             "example": "How to improve communication with my partner"
         },
         {
             "id": "job",
             "name": "Job Finder",
             "description": "Suggests career paths and job opportunities",
             "example": "What jobs are good for someone with creative skills?"
         },
         {
             "id": "gardening",
             "name": "Gardening Expert",
             "description": "Offers tips for growing plants",
             "example": "How to grow herbs indoors"
         },
         {
             "id": "parenting",
             "name": "Parenting Coach",
             "description": "Provides advice for parents",
             "example": "Tips for managing toddler tantrums"
         },
         {
             "id": "resume_review",
             "name": "Resume Reviewer",
             "description": "Reviews and improves existing resumes",
             "example": "Review my resume and suggest improvements"
         },
         {
             "id": "horoscope",
             "name": "Horoscope Reader",
             "description": "Provides zodiac insights",
             "example": "What's the horoscope for Taurus today?"
         },
         {
             "id": "event",
             "name": "Event Planner",
             "description": "Helps plan and organize events",
             "example": "Help me plan a wedding anniversary party"
         },
         {
             "id": "shopping",
             "name": "Shopping Assistant",
             "description": "Helps find products and deals online",
             "example": "Find a good laptop under $1000"
         },
         {
             "id": "mental_health",
             "name": "Mental Health Support",
             "description": "Offers mindfulness and stress management tips",
             "example": "How to manage work-related anxiety"
         },
         {
             "id": "productivity",
             "name": "Productivity Coach",
             "description": "Provides strategies to improve focus and efficiency",
             "example": "Tips to stop procrastinating on projects"
         },
         {
             "id": "budget",
             "name": "Budget Tracker",
             "description": "Helps monitor and optimize expenses",
             "example": "How to track and reduce my monthly expenses"
         },
         {
             "id": "affirmation",
             "name": "Affirmation Agent",
             "description": "Generates empowering daily affirmations",
             "example": "Give me affirmations for confidence"
         },
         {
             "id": "nutrition",
             "name": "Nutritionist",
             "description": "Provides balanced dietary advice",
             "example": "Create a diet plan for weight gain"
         },
         {
             "id": "pet",
             "name": "Pet Care Advisor",
             "description": "Offers pet health and training advice",
             "example": "How to train a new puppy"
         },
         {
             "id": "decor",
             "name": "Home Decor Advisor",
             "description": "Suggests interior design ideas",
             "example": "Ideas for decorating a small apartment"
         },
         {
             "id": "sleep",
             "name": "Sleep Coach",
             "description": "Provides tips for better sleep",
             "example": "How to improve sleep quality for insomnia"
         },
         {
             "id": "tech",
             "name": "Tech Support",
             "description": "Troubleshoots common technology problems",
             "example": "My Wi-Fi keeps disconnecting. How can I fix it?"
         },
         {
             "id": "time",
             "name": "Time Management Coach",
             "description": "Helps schedule and prioritize tasks",
             "example": "How to manage time better as a student"
         },
         {
             "id": "career",
             "name": "Career Counselor",
             "description": "Advises on career growth and transitions",
             "example": "How to switch from marketing to product management"
         },
         {
             "id": "academic",
             "name": "Academic Tutor",
             "description": "Explains academic concepts",
             "example": "Explain how photosynthesis works"
         },
         {
            "id": "festival",
            "name": "Festival Finder",
            "description": "Discovers local and global cultural events",
            "example": "Find music festivals in Europe this summer"
            },
        {
            "id": "resume_keywords",
            "name": "Resume Keyword Optimizer",
            "description": "Enhances resumes with ATS-friendly keywords",
            "example": "Add relevant keywords to my software engineer resume"
        },
        {
            "id": "cover_letter",
            "name": "Cover Letter Builder",
            "description": "Creates personalized cover letters",
            "example": "Write a cover letter for a marketing manager position"
        },
        {     
                "id": "voice",
                "name": "Voice Tone Coach",
                "description": "Improves speaking confidence and delivery",
                "example": "How to sound more confident during presentations"
        },
        {
                "id": "crypto",
                "name": "Crypto Advisor",
                "description": "Explains cryptocurrency concepts",
                "example": "Explain the difference between Bitcoin and Ethereum"
        },
        {
                "id": "loan",
                "name": "Loan Calculator",
                "description": "Calculates loan repayment options",
                "example": "Calculate EMI for a home loan of $200,000 for 20 years"
        },
        { 
                "id": "insurance",
                "name": "Insurance Helper",
                "description": "Explains insurance options and terms",
                "example": "What should I look for in a health insurance plan?"
        },
        {
                "id": "car",
                "name": "Car Advisor",
                "description": "Helps with car buying and selling decisions",
                "example": "Recommend cars under $20,000 with good fuel efficiency"
        },
        {
                "id": "appliance",
                "name": "Appliance Recommender",
                "description": "Suggests home and kitchen appliances",
                "example": "Recommend an energy-efficient refrigerator"
        },
        {
                "id": "minimalism",
                "name": "Minimalism Coach",
                "description": "Provides tips for decluttering and simplifying life",
                "example": "How to declutter and organize my living room"
        },
        {
                "id": "coding_prep",
                "name": "Coding Interview Prep",
                "description": "Helps prepare for technical interviews",
                "example": "Practice questions on binary trees for coding interviews"
        },
        {
                "id": "habit",
                "name": "Habit Tracker",
                "description": "Helps build and track positive habits",
                "example": "How to build a consistent morning routine"
        },
        {
                "id": "fashion",
                "name": "Fashion Stylist",
                "description": "Recommends outfit ideas and style tips",
                "example": "What should I wear to a formal dinner event?"
        },
        {
                "id": "music",
                "name": "Music Recommender",
                "description": "Suggests music based on preferences",
                "example": "Recommend chill evening playlist songs"
         }
]

    
    return jsonify({
        'status': 'success',
        'agents': agents
    })

@app.route('/api/tickets/<booking_id>', methods=['GET'])
def get_ticket(booking_id):
    """Endpoint to download a ticket"""
    try:
        ticket_info = booking_system.get_ticket(booking_id)
        return send_file(
            ticket_info['path'],
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{ticket_info['type']}_ticket_{booking_id}.pdf"
        )
    except Exception as e:
        logger.error(f"Error retrieving ticket: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"Error retrieving ticket: {str(e)}"
        }), 404

if __name__ == '__main__':
    # Ensure tickets directory exists
    os.makedirs('tickets', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)