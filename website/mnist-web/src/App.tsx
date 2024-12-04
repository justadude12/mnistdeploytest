import React, { useRef, useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { Box, Button, Typography, Paper } from '@mui/material';
import axios from 'axios';

interface Prediction {
  digit: number;
  probability: number;
}

const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const session = await ort.InferenceSession.create('/model.onnx');
        setSession(session);
      } catch (e) {
        console.error('Failed to load model:', e);
        throw e;
      }
    };
    loadModel();
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      ctx.beginPath();
      ctx.moveTo(
        e.clientX - canvas.offsetLeft,
        e.clientY - canvas.offsetTop
      );
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      ctx.lineWidth = 13;
      ctx.lineCap = 'round';
      ctx.lineTo(
        e.clientX - canvas.offsetLeft,
        e.clientY - canvas.offsetTop
      );
      ctx.stroke();
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setPredictions([]);
    }
  };

  const preprocessCanvas = (canvas: HTMLCanvasElement): Float32Array => {
    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) throw new Error('Could not get canvas context');
    
    // Draw the original canvas content onto the smaller canvas
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data and normalize
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = new Float32Array(28 * 28);
    
    // Convert to grayscale and normalize to [-1, 1]
    for (let i = 0; i < imageData.data.length; i += 4) {
      const grayscale = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
      data[i / 4] = (255 - grayscale) / 255 * 2 - 1; // Invert and normalize
    }
    
    return data;
  };

  const predict = async () => {
    if (!session || !canvasRef.current) return;
    
    try {
      // Preprocess the canvas image
      const data = preprocessCanvas(canvasRef.current);
      
      // Create tensor with correct shape [1, 1, 28, 28]
      const inputTensor = new ort.Tensor('float32', data, [1, 1, 28, 28]);
      
      // Run inference
      const outputs = await session.run({
        input: inputTensor
      });
      
      // Get the output data
      const outputData = outputs.output.data as Float32Array;
      
      // Convert to probabilities using softmax
      const softmaxOutput = softmax(Array.from(outputData));
      
      // Create predictions array
      const predictions = softmaxOutput.map((prob, index) => ({
        digit: index,
        probability: prob
      })).sort((a, b) => b.probability - a.probability);
      
      setPredictions(predictions);
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  const saveImageToGitHub = async (imageDataUrl: string, fileName: string) => {
    const token = process.env.REACT_APP_GITHUB_TOKEN;
    const repo = 'justadude12/mnist-images';
    const path = `${fileName}.png`;
    const content = imageDataUrl.split(',')[1];

    try {
      const response = await axios.put(
        `https://api.github.com/repos/${repo}/contents/${path}`,
        {
          message: `Add image ${fileName}`,
          content: content,
        },
        { headers: { Authorization: `token ${token}`,
          Accept: 'application/vnd.github.v3+json',
         },
        }
      );
      console.log('Image saved to GitHub:', response.data.content.html_url);
    } catch (error) {
      console.error('Error saving image to GitHub:', error);
    }
  };

  const saveImage = async () => {
    if (!canvasRef.current) return;
  
    try {
      const dataUrl = canvasRef.current.toDataURL('image/png');
      const timestamp = Date.now();
      const fileName = `digit_9_${timestamp}`;
  
      await saveImageToGitHub(dataUrl, fileName);
    } catch (error) {
      console.error('Error saving image:', error);
    }
  };

  // Helper function for softmax
  const softmax = (arr: number[]): number[] => {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b);
    return exp.map(x => x / sum);
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        MNIST Digit Classifier
      </Typography>
      
      <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          style={{ border: '1px solid black' }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
        />
        
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={predict} sx={{ mr: 1 }}>
            Predict
          </Button>
          <Button variant="outlined" onClick={clearCanvas}>
            Clear
          </Button>
          <Button
            variant="contained"
            onClick={saveImage}
            sx = {{mr : 1}}
            color="secondary">
            Save as 9
          </Button>
        </Box>
      </Paper>

      {predictions.length > 0 && (
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Predictions
          </Typography>
          {predictions.slice(0, 3).map(({ digit, probability }) => (
            <Typography key={digit}>
              Digit {digit}: {(probability * 100).toFixed(2)}%
            </Typography>
          ))}
        </Paper>
      )}
    </Box>
  );
};

export default App;