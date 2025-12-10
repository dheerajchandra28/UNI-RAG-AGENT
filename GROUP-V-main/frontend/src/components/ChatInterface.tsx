import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send } from 'lucide-react';
import { Message } from './Message';
import { LoadingOrb } from './LoadingOrb';
import axios from 'axios';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  source?: string;
  audioUrl?: string;
}

interface ChatInterfaceProps {
  mode: 'folder' | 'file';
  selectedFile: File | null;
}

const API_BASE_URL = 'http://127.0.0.1:8000';

export const ChatInterface = ({ mode, selectedFile }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    if (mode === 'file' && !selectedFile) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let response;

      if (mode === 'folder') {
        response = await axios.post(`${API_BASE_URL}/chat-folder`, {
          query: input,
        });
      } else {
        const formData = new FormData();
        formData.append('file', selectedFile!);
        formData.append('query', input);

        response = await axios.post(`${API_BASE_URL}/chat-file`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.data.response,
        source: response.data.source,
        audioUrl: response.data.audio_url,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please make sure the API server is running at http://127.0.0.1:8000',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isDisabled = mode === 'file' && !selectedFile;

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center text-muted-foreground mt-20"
          >
            <h2 className="text-2xl font-bold mb-2 text-glow text-primary">
              Advanced RAG Agent
            </h2>
            <p>
              {mode === 'folder'
                ? 'Ask me anything about the knowledge base'
                : 'Upload a document and ask questions about it'}
            </p>
          </motion.div>
        )}

        {messages.map((message, index) => (
          <Message key={index} {...message} />
        ))}

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="max-w-[70%]">
              <LoadingOrb />
              <p className="text-center text-muted-foreground text-sm">Thinking...</p>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-6 border-t border-primary/20 backdrop-cyber">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isDisabled || isLoading}
            placeholder={
              isDisabled
                ? 'Please upload a file first...'
                : 'Type your message...'
            }
            className="flex-1 px-6 py-4 bg-input border border-primary/30 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <motion.button
            onClick={handleSend}
            disabled={isDisabled || isLoading || !input.trim()}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-4 bg-gradient-primary rounded-lg font-semibold text-primary-foreground disabled:opacity-50 disabled:cursor-not-allowed glow-cyan transition-all"
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </div>
  );
};
