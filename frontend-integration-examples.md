# Frontend Integration Examples

This guide shows how to integrate the Event Recommendation Chatbot into your web application.

## API Endpoints

### Base URL
```
http://your-api-server:8000
```

### Main Endpoints
- `POST /api/chat` - Main chat with user context
- `GET /api/user/preferences/{user_id}` - Check user preferences
- `POST /api/user/preferences` - Save user preferences
- `WS /ws/chat` - WebSocket real-time chat

## JavaScript Integration

### 1. Basic Chat Integration

```javascript
class EventChatbot {
    constructor(apiBaseUrl, userId, userCity) {
        this.apiBaseUrl = apiBaseUrl;
        this.userId = userId;
        this.userCity = userCity;
    }

    async sendMessage(message) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.userId,
                    user_current_city: this.userCity
                })
            });

            const data = await response.json();
            
            // Handle preference collection
            if (data.needs_preferences) {
                await this.handlePreferenceCollection();
                // Retry original message after collecting preferences
                return this.sendMessage(message);
            }

            return data;
        } catch (error) {
            console.error('Chat error:', error);
            return {
                success: false,
                message: 'Sorry, I encountered an error. Please try again.'
            };
        }
    }

    async handlePreferenceCollection() {
        // Show preference collection UI
        const preferences = await this.showPreferenceDialog();
        
        if (preferences) {
            await this.savePreferences(preferences);
        }
    }

    async savePreferences(preferences) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/user/preferences`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    activities: preferences.activities,
                    preferred_locations: preferences.locations,
                    preferred_time: preferences.time,
                    budget_range: preferences.budget
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Error saving preferences:', error);
        }
    }

    async showPreferenceDialog() {
        // Example implementation - customize for your UI
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.innerHTML = `
                <div class="preference-modal">
                    <h3>Tell us your preferences</h3>
                    <label>
                        Activities (select multiple):
                        <select multiple id="activities">
                            <option value="sports">Sports</option>
                            <option value="tech">Technology</option>
                            <option value="music">Music</option>
                            <option value="arts">Arts</option>
                            <option value="networking">Networking</option>
                        </select>
                    </label>
                    <label>
                        Preferred Areas:
                        <input type="text" id="locations" placeholder="e.g., Downtown, North Zone">
                    </label>
                    <label>
                        Preferred Time:
                        <select id="time">
                            <option value="morning">Morning</option>
                            <option value="evening">Evening</option>
                            <option value="weekend">Weekend</option>
                        </select>
                    </label>
                    <button onclick="submitPreferences()">Save Preferences</button>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            window.submitPreferences = () => {
                const activities = Array.from(document.getElementById('activities').selectedOptions)
                    .map(option => option.value);
                const locations = document.getElementById('locations').value.split(',').map(s => s.trim());
                const time = document.getElementById('time').value;
                
                document.body.removeChild(modal);
                resolve({
                    activities,
                    locations,
                    time,
                    budget: null
                });
            };
        });
    }
}

// Usage Example
const chatbot = new EventChatbot('http://localhost:8000', 'user123', 'Mumbai');

// Send a message
chatbot.sendMessage('find events for me').then(response => {
    console.log('Bot response:', response.message);
    if (response.events && response.events.length > 0) {
        console.log('Recommended events:', response.events);
    }
});
```

### 2. React Component Example

```jsx
import React, { useState, useEffect } from 'react';

const ChatWidget = ({ userId, userCity, apiBaseUrl }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [showPreferences, setShowPreferences] = useState(false);

    const sendMessage = async (text) => {
        setLoading(true);
        
        // Add user message to chat
        const userMessage = { type: 'user', text, timestamp: new Date() };
        setMessages(prev => [...prev, userMessage]);

        try {
            const response = await fetch(`${apiBaseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    user_id: userId,
                    user_current_city: userCity
                })
            });

            const data = await response.json();

            if (data.needs_preferences) {
                setShowPreferences(true);
                const botMessage = { 
                    type: 'bot', 
                    text: data.message, 
                    timestamp: new Date(),
                    needsPreferences: true
                };
                setMessages(prev => [...prev, botMessage]);
            } else {
                const botMessage = { 
                    type: 'bot', 
                    text: data.message, 
                    events: data.events,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, botMessage]);
            }
        } catch (error) {
            const errorMessage = { 
                type: 'bot', 
                text: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        }

        setLoading(false);
        setInput('');
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim()) {
            sendMessage(input.trim());
        }
    };

    const savePreferences = async (preferences) => {
        try {
            await fetch(`${apiBaseUrl}/api/user/preferences`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    ...preferences
                })
            });
            
            setShowPreferences(false);
            // Retry last user message
            const lastUserMessage = messages.filter(m => m.type === 'user').pop();
            if (lastUserMessage) {
                sendMessage(lastUserMessage.text);
            }
        } catch (error) {
            console.error('Error saving preferences:', error);
        }
    };

    return (
        <div className="chat-widget">
            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        <div className="message-text">{msg.text}</div>
                        {msg.events && msg.events.length > 0 && (
                            <div className="events-list">
                                {msg.events.map((event, i) => (
                                    <div key={i} className="event-card">
                                        <h4>{event.name}</h4>
                                        <p>{event.club_name}</p>
                                        <p>{event.location.city}</p>
                                        <p>₹{event.price}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
                {loading && <div className="message bot">Thinking...</div>}
            </div>

            <form onSubmit={handleSubmit} className="chat-input">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about events..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading || !input.trim()}>
                    Send
                </button>
            </form>

            {showPreferences && (
                <PreferenceModal 
                    onSave={savePreferences}
                    onCancel={() => setShowPreferences(false)}
                />
            )}
        </div>
    );
};

const PreferenceModal = ({ onSave, onCancel }) => {
    const [activities, setActivities] = useState([]);
    const [locations, setLocations] = useState('');
    const [time, setTime] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        onSave({
            activities,
            preferred_locations: locations.split(',').map(s => s.trim()),
            preferred_time: time
        });
    };

    return (
        <div className="preference-modal">
            <div className="modal-content">
                <h3>Tell us your preferences</h3>
                <form onSubmit={handleSubmit}>
                    <div>
                        <label>Activities you enjoy:</label>
                        {['sports', 'tech', 'music', 'arts', 'networking'].map(activity => (
                            <label key={activity}>
                                <input
                                    type="checkbox"
                                    checked={activities.includes(activity)}
                                    onChange={(e) => {
                                        if (e.target.checked) {
                                            setActivities([...activities, activity]);
                                        } else {
                                            setActivities(activities.filter(a => a !== activity));
                                        }
                                    }}
                                />
                                {activity}
                            </label>
                        ))}
                    </div>
                    
                    <div>
                        <label>
                            Preferred Areas (comma-separated):
                            <input
                                type="text"
                                value={locations}
                                onChange={(e) => setLocations(e.target.value)}
                                placeholder="Downtown, North Zone"
                            />
                        </label>
                    </div>
                    
                    <div>
                        <label>
                            Preferred Time:
                            <select value={time} onChange={(e) => setTime(e.target.value)}>
                                <option value="">Select time</option>
                                <option value="morning">Morning</option>
                                <option value="evening">Evening</option>
                                <option value="weekend">Weekend</option>
                            </select>
                        </label>
                    </div>
                    
                    <div className="modal-buttons">
                        <button type="submit">Save Preferences</button>
                        <button type="button" onClick={onCancel}>Cancel</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ChatWidget;
```

### 3. WebSocket Real-time Chat

```javascript
class RealtimeChatbot {
    constructor(wsUrl, userId, userCity) {
        this.wsUrl = wsUrl;
        this.userId = userId;
        this.userCity = userCity;
        this.ws = null;
        this.messageCallbacks = [];
    }

    connect() {
        this.ws = new WebSocket(this.wsUrl);
        
        this.ws.onopen = () => {
            console.log('Connected to chatbot');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.messageCallbacks.forEach(callback => callback(data));
        };

        this.ws.onclose = () => {
            console.log('Disconnected from chatbot');
            // Reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                message,
                user_id: this.userId,
                user_current_city: this.userCity
            }));
        }
    }

    onMessage(callback) {
        this.messageCallbacks.push(callback);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage
const realtimeChat = new RealtimeChatbot('ws://localhost:8000/ws/chat', 'user123', 'Mumbai');

realtimeChat.onMessage((data) => {
    if (data.type === 'response') {
        console.log('Bot response:', data.message);
        if (data.recommendations && data.recommendations.length > 0) {
            console.log('Events:', data.recommendations);
        }
    }
});

realtimeChat.connect();

// Send message
realtimeChat.sendMessage('find tech events this weekend');
```

## Integration into Your Application

### 1. Add to Existing User Session

```javascript
// When user logs in, initialize chatbot with their context
function initializeChatbot(user) {
    const chatbot = new EventChatbot(
        'http://your-api-server:8000',
        user.id,
        user.currentCity
    );
    
    // Store in global scope or state management
    window.eventChatbot = chatbot;
}

// Chat button click handler
async function openChat() {
    const message = prompt('What events are you looking for?');
    if (message) {
        const response = await window.eventChatbot.sendMessage(message);
        showChatResponse(response);
    }
}
```

### 2. CSS Styling

```css
.chat-widget {
    width: 400px;
    height: 500px;
    border: 1px solid #ddd;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.message {
    margin-bottom: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 80%;
}

.message.user {
    background: #007bff;
    color: white;
    margin-left: auto;
}

.message.bot {
    background: #f1f1f1;
    color: #333;
}

.events-list {
    margin-top: 12px;
}

.event-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 8px;
}

.chat-input {
    display: flex;
    padding: 16px;
    border-top: 1px solid #ddd;
}

.chat-input input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin-right: 8px;
}

.chat-input button {
    padding: 8px 16px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.preference-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-content {
    background: white;
    padding: 24px;
    border-radius: 8px;
    width: 400px;
    max-width: 90vw;
}
```

This integration guide provides everything needed to add the chatbot to your application with automatic user context handling and preference management.