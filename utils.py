import logging
import os
import json
from datetime import datetime

def setup_logging():
    """Configure logging with timestamp and log level"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_api_keys():
    """Validate that all required API keys are present"""
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENWEATHER_API_KEY": os.getenv("OPENWEATHER_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        logging.warning(f"Missing API keys: {', '.join(missing_keys)}")
        return False
    
    return True

def save_conversation(user_id, prompt, response, agent_type):
    """Save conversation history to a JSON file"""
    try:
        history_dir = "conversation_history"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        history_file = os.path.join(history_dir, f"{user_id}.json")
        
        # Create new history if file doesn't exist
        if not os.path.exists(history_file):
            history = []
        else:
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Add new conversation entry
        history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "agent_type": agent_type
        })
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        return True
    except Exception as e:
        logging.error(f"Error saving conversation: {str(e)}")
        return False