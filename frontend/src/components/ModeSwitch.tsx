import { motion } from 'framer-motion';

type Mode = 'folder' | 'file';

interface ModeSwitchProps {
  activeMode: Mode;
  onModeChange: (mode: Mode) => void;
}

export const ModeSwitch = ({ activeMode, onModeChange }: ModeSwitchProps) => {
  return (
    <div className="flex gap-2 p-1 bg-card rounded-lg border border-primary/20 w-fit mx-auto mb-8">
      <motion.button
        onClick={() => onModeChange('folder')}
        className={`relative px-6 py-3 rounded-md font-medium transition-colors ${
          activeMode === 'folder' ? 'text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {activeMode === 'folder' && (
          <motion.div
            layoutId="activeTab"
            className="absolute inset-0 bg-gradient-primary rounded-md glow-cyan"
            transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
          />
        )}
        <span className="relative z-10">Chat with Knowledge Base</span>
      </motion.button>

      <motion.button
        onClick={() => onModeChange('file')}
        className={`relative px-6 py-3 rounded-md font-medium transition-colors ${
          activeMode === 'file' ? 'text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {activeMode === 'file' && (
          <motion.div
            layoutId="activeTab"
            className="absolute inset-0 bg-gradient-primary rounded-md glow-cyan"
            transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
          />
        )}
        <span className="relative z-10">Upload Document</span>
      </motion.button>
    </div>
  );
};
