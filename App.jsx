import React, { useState, useEffect } from 'react';
import './App.css';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [conversations, setConversations] = useState([
    {
      id: 'welcome',
      messages: [
        {
          id: 'welcome-msg',
          role: 'assistant',
          content: 'Welcome to Multi-Agent Chat! How can I help you today?',
          timestamp: new Date().toISOString(),
          agentType: 'system'
        }
      ]
    }
  ]);
  const [activeConversation, setActiveConversation] = useState('welcome');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Fetch available agents on component mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/agents`);
        if (response.data.status === 'success') {
          setAgents(response.data.agents);
        }
      } catch (error) {
        console.error('Error fetching agents:', error);
        toast.error('Failed to fetch available agents');
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  // Create a new conversation
  const createNewConversation = () => {
    const id = `conv-${Date.now()}`;
    const newConversation = {
      id,
      messages: []
    };

    setConversations([...conversations, newConversation]);
    setActiveConversation(id);
  };

  // Send a message to the backend
  const sendMessage = async (message) => {
    // Add user message to conversation
    const messageId = `msg-${Date.now()}`;
    const userMessage = {
      id: messageId,
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };

    // Update conversation with user message
    const updatedConversations = conversations.map(conv => {
      if (conv.id === activeConversation) {
        return {
          ...conv,
          messages: [...conv.messages, userMessage]
        };
      }
      return conv;
    });
    setConversations(updatedConversations);

    // Add thinking message
    const thinkingMessageId = `thinking-${Date.now()}`;
    const thinkingMessage = {
      id: thinkingMessageId,
      role: 'assistant',
      content: 'Thinking...',
      timestamp: new Date().toISOString(),
      isThinking: true
    };

    const withThinkingMessage = updatedConversations.map(conv => {
      if (conv.id === activeConversation) {
        return {
          ...conv,
          messages: [...conv.messages, thinkingMessage]
        };
      }
      return conv;
    });
    setConversations(withThinkingMessage);

    try {
      // Send message to API
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: message
      });

      if (response.data.status === 'success') {
        // Replace thinking message with actual response
        const assistantMessage = {
          id: `resp-${Date.now()}`,
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date().toISOString(),
          agentType: response.data.agent_type,
          processingTime: response.data.processing_time
        };

        const finalConversations = withThinkingMessage.map(conv => {
          if (conv.id === activeConversation) {
            return {
              ...conv,
              messages: conv.messages
                .filter(msg => msg.id !== thinkingMessageId)
                .concat(assistantMessage)
            };
          }
          return conv;
        });
        setConversations(finalConversations);
      } else {
        throw new Error(response.data.message || 'Unknown error');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Replace thinking message with error
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error.message || 'Failed to get response from server'}`,
        timestamp: new Date().toISOString(),
        isError: true
      };

      const errorConversations = withThinkingMessage.map(conv => {
        if (conv.id === activeConversation) {
          return {
            ...conv,
            messages: conv.messages
              .filter(msg => msg.id !== thinkingMessageId)
              .concat(errorMessage)
          };
        }
        return conv;
      });
      setConversations(errorConversations);
      
      toast.error('Failed to process your message');
    }
  };

  return (
    <div className="App h-screen flex flex-col">
      <ToastContainer position="top-right" autoClose={3000} />
      <Header 
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} 
        createNewConversation={createNewConversation}
      />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          isOpen={isSidebarOpen}
          conversations={conversations}
          activeConversation={activeConversation}
          setActiveConversation={setActiveConversation}
          agents={agents}
          loading={loading}
        />
        <ChatInterface 
          conversation={conversations.find(conv => conv.id === activeConversation)}
          sendMessage={sendMessage}
          agents={agents}
        />
      </div>
    </div>
  );
}

export default App;