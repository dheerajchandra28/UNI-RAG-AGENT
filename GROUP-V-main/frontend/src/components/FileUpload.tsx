import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, File, X } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
}

export const FileUpload = ({ onFileSelect, selectedFile }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      
      const file = e.dataTransfer.files[0];
      if (file && (file.type === 'application/pdf' || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const clearFile = useCallback(() => {
    onFileSelect(null);
  }, [onFileSelect]);

  return (
    <div className="mb-6">
      {!selectedFile ? (
        <motion.div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-primary bg-primary/10 glow-cyan'
              : 'border-primary/30 hover:border-primary/60 hover:bg-primary/5'
          }`}
          whileHover={{ scale: 1.01 }}
          animate={isDragging ? { scale: 1.02 } : { scale: 1 }}
        >
          <input
            type="file"
            accept=".pdf,.docx"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <motion.div
            animate={isDragging ? { y: -10 } : { y: 0 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-primary" />
            <h3 className="text-xl font-semibold mb-2 text-foreground">
              {isDragging ? 'Drop your file here' : 'Upload a Document'}
            </h3>
            <p className="text-muted-foreground">
              Drag and drop a PDF or DOCX file, or click to browse
            </p>
          </motion.div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-4 p-4 bg-card border border-primary/30 rounded-lg"
        >
          <File className="w-8 h-8 text-primary" />
          <div className="flex-1">
            <p className="font-medium text-foreground">{selectedFile.name}</p>
            <p className="text-sm text-muted-foreground">
              {(selectedFile.size / 1024).toFixed(2)} KB
            </p>
          </div>
          <button
            onClick={clearFile}
            className="p-2 hover:bg-destructive/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-destructive" />
          </button>
        </motion.div>
      )}
    </div>
  );
};
