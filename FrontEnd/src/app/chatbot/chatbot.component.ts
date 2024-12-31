import { Component, OnInit } from '@angular/core';
import { ChatbotService } from '../chatbot.service';

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.scss']
})
export class ChatbotComponent implements OnInit {
  messages: { sender: string, message: string }[] = [];
  userMessage: string = '';

  constructor(private chatbotService: ChatbotService) {}

  ngOnInit(): void {}

  sendMessage(): void {
    if (this.userMessage.trim()) {
      // Push user's message to the chat
      this.messages.push({ sender: 'user', message: this.userMessage });
      
      // Send the user's query to the chatbot service
      this.chatbotService.sendQuery(this.userMessage).subscribe(
        (response) => {
          // Assuming the response is an object with a `response` key
          const chatbotResponse = response?.response?.response || 'No response available';

          // Push the bot's response to the chat
          this.messages.push({ sender: 'bot', message: chatbotResponse });
          this.userMessage = '';  // Clear input field after sending
        },
        (error) => {
          console.error('Error:', error);
          this.messages.push({ sender: 'bot', message: 'Sorry, something went wrong.' });
        }
      );
    }
  }
}
