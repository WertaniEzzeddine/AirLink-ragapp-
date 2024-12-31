// src/app/chatbot.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatbotService {
  private apiUrl = 'http://localhost:8000/chatbot/';  // Replace with your FastAPI URL

  constructor(private http: HttpClient) {}

  sendQuery(query: string): Observable<any> {
    return this.http.post<any>(this.apiUrl, { query });
  }
}
