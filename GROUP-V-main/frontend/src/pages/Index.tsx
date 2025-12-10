import { useState } from 'react';
import { motion } from 'framer-motion';
import { Background3D } from '@/components/Background3D';
import { ModeSwitch } from '@/components/ModeSwitch';
import { FileUpload } from '@/components/FileUpload';
import { ChatInterface } from '@/components/ChatInterface';

type Mode = 'folder' | 'file';

const Index = () => {
  const [mode, setMode] = useState<Mode>('folder');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    setSelectedFile(null);
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      <Background3D />
      
      <div className="relative z-10 min-h-screen flex flex-col">
        <header className="backdrop-cyber border-b border-primary/20">
          <div className="container mx-auto px-6 py-6">
            <motion.h1
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-4xl font-bold text-center text-glow"
            >
              <span className="text-primary text-glow">
                  Deep Search
              </span>
            </motion.h1>
          </div>
        </header>

        <main className="flex-1 container mx-auto px-6 py-8 flex flex-col max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <ModeSwitch activeMode={mode} onModeChange={handleModeChange} />
          </motion.div>

          <motion.div
            key={mode}
            initial={{ opacity: 0, x: mode === 'folder' ? -20 : 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: mode === 'folder' ? 20 : -20 }}
            transition={{ duration: 0.3 }}
            className="flex-1 flex flex-col"
          >
            {mode === 'file' && (
              <FileUpload
                selectedFile={selectedFile}
                onFileSelect={setSelectedFile}
              />
            )}

            <div className="flex-1 backdrop-cyber rounded-lg border border-primary/20 overflow-hidden flex flex-col">
              <ChatInterface mode={mode} selectedFile={selectedFile} />
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default Index;
