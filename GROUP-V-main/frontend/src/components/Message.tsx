import { motion } from 'framer-motion';
import { User, Bot, FileText, Volume2 } from 'lucide-react';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  source?: string;
  audioUrl?: string;
}

export const Message = ({ role, content, source, audioUrl }: MessageProps) => {
  const isUser = role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-4 mb-6 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-gradient-primary glow-cyan' : 'bg-secondary/20 border-2 border-secondary glow-purple'
        }`}
      >
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5 text-secondary" />}
      </div>

      <div className={`flex flex-col gap-2 max-w-[70%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-4 py-3 rounded-lg backdrop-cyber border ${
            isUser
              ? 'bg-primary/10 border-primary/30'
              : 'bg-card border-secondary/30'
          }`}
        >
          <p className="text-foreground whitespace-pre-wrap">{content}</p>
        </div>

        {source && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 rounded-md text-sm text-muted-foreground border border-primary/20"
          >
            <FileText className="w-4 h-4" />
            <span>Source: {source}</span>
          </motion.div>
        )}

        {audioUrl && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3 px-4 py-2 bg-card border border-primary/30 rounded-lg w-full"
          >
            <Volume2 className="w-5 h-5 text-primary" />
            <audio controls className="flex-1 h-8" style={{ filter: 'hue-rotate(180deg)' }}>
              <source src={audioUrl} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};
