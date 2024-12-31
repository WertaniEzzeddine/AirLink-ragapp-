// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { AppRoutingModule } from './app-routing.module'; // Import AppRoutingModule

@NgModule({
  declarations: [
    AppComponent,
    ChatbotComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    AppRoutingModule  // Add AppRoutingModule here
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
