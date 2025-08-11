css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 12px; 
    margin-bottom: 1.2rem; 
    display: flex;
    max-width: 85%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.chat-message.user {
    background-color: #007bff;
    margin-left: auto;
    margin-right: 0;
    color: white;
    border-bottom-right-radius: 4px;
}
.chat-message.bot {
    background-color: #f8f9fa;
    color: #212529;
    border: 1px solid #e9ecef;
    margin-left: 0;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}
.chat-message .avatar {
    width: 45px;
    height: 45px;
    margin-right: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-message.user .avatar {
    margin-right: 0;
    margin-left: 12px;
    order: 2;
}
.chat-message .avatar img {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    flex: 1;
    line-height: 1.5;
    font-size: 15px;
    display: flex;
    align-items: center;
}
.chat-message.user .message {
    order: 1;
    justify-content: flex-end;
    text-align: right;
}

/* Smooth animations */
.chat-message {
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/BZhxXpS/ai-icon.png" alt="Assistant">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/Yy7KTkS/person-icon.png" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
