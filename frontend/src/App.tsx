import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { 
  Settings, 
  Languages, 
  Trash2, 
  BookOpen, 
  Volume2
} from 'lucide-react';

const ASL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const ISL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const App: React.FC = () => {
  const [mode, setMode] = useState<'ASL' | 'ISL'>('ASL');
  const [prediction, setPrediction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [sentence, setSentence] = useState<string>('');
  const [translatedText, setTranslatedText] = useState<string>('');
  const [targetLang, setTargetLang] = useState<string>('hi');
  const [isTeaching, setIsTeaching] = useState<boolean>(false);
  const [targetLetter, setTargetLetter] = useState<string>('A');
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  
  const webcamRef = useRef<Webcam>(null);
  const ws = useRef<WebSocket | null>(null);

  // Initialize WebSocket
  useEffect(() => {
    const connect = () => {
      setWsStatus('connecting');
      ws.current = new WebSocket('ws://localhost:8000/ws/predict');
      
      ws.current.onopen = () => setWsStatus('connected');
      ws.current.onerror = () => setWsStatus('error');
      ws.current.onclose = () => {
        setWsStatus('connecting');
        setTimeout(connect, 3000); // Reconnect after 3s
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setPrediction(data.letter || '');
        setConfidence(data.confidence || 0);
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, []);

  // Frame Capture and Sending
  const capture = useCallback(() => {
    if (webcamRef.current && ws.current?.readyState === WebSocket.OPEN) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        ws.current.send(JSON.stringify({
          image: imageSrc,
          mode: mode
        }));
      }
    }
  }, [mode]);

  useEffect(() => {
    const interval = setInterval(capture, 100); // 10 FPS
    return () => clearInterval(interval);
  }, [capture]);

  // Key Controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        if (prediction) {
          setSentence(prev => prev + prediction);
        }
      } else if (e.code === 'Backspace') {
        setSentence(prev => prev.slice(0, -1));
      } else if (e.code === 'Enter') {
        setSentence(prev => prev + ' ');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [prediction]);

  // Translation
  const translate = async () => {
    if (!sentence) return;
    try {
      const response = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentence, target_lang: targetLang })
      });
      const data = await response.json();
      setTranslatedText(data.translated);
    } catch (error) {
      console.error("Translation error:", error);
    }
  };

  // Text to Speech
  const speak = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    // You can customize voice and rate here
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      <header className="flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Languages className="text-blue-400" />
          Multilingual Sign-to-Speech
        </h1>
        <div className="flex gap-4">
          <button 
            onClick={() => setIsTeaching(!isTeaching)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${isTeaching ? 'bg-blue-600' : 'bg-slate-700 hover:bg-slate-600'}`}
          >
            <BookOpen size={20} />
            Teaching Module
          </button>
          <div className="flex bg-slate-800 rounded-lg p-1">
            <button 
              onClick={() => setMode('ASL')}
              className={`px-4 py-1 rounded-md transition ${mode === 'ASL' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
            >
              ASL
            </button>
            <button 
              onClick={() => setMode('ISL')}
              className={`px-4 py-1 rounded-md transition ${mode === 'ISL' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
            >
              ISL
            </button>
          </div>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Camera Section */}
        <div className="relative rounded-2xl overflow-hidden bg-black aspect-video border-4 border-slate-800 shadow-2xl">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="w-full h-full object-cover"
            videoConstraints={{ 
              width: 640,
              height: 480,
              facingMode: "user" 
            }}
          />
          <div className="absolute top-4 left-4 flex flex-col gap-2">
            <div className="bg-black/50 backdrop-blur-md px-4 py-2 rounded-full flex items-center gap-2 border border-white/20">
              <div className={`w-3 h-3 rounded-full ${prediction ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
              <span className="text-sm font-medium">{mode} Mode Active</span>
            </div>
            <div className="bg-black/50 backdrop-blur-md px-4 py-1 rounded-full flex items-center gap-2 border border-white/20 w-fit">
              <div className={`w-2 h-2 rounded-full ${wsStatus === 'connected' ? 'bg-green-400' : wsStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'}`} />
              <span className="text-[10px] uppercase font-bold tracking-tighter">
                {wsStatus === 'connected' ? 'Server Connected' : wsStatus === 'error' ? 'Server Error' : 'Connecting...'}
              </span>
            </div>
          </div>
          
          {prediction && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="bg-white/10 backdrop-blur-sm border-2 border-white/30 rounded-3xl p-8 flex flex-col items-center">
                <span className="text-8xl font-black text-white drop-shadow-lg">{prediction}</span>
                <span className="text-sm font-bold text-blue-300 mt-2">{(confidence * 100).toFixed(1)}% Match</span>
              </div>
            </div>
          )}
        </div>

        {/* Controls & Results */}
        <div className="flex flex-col gap-6">
          {/* Main Output */}
          <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-slate-400 font-bold uppercase text-xs tracking-widest">Constructed Sentence</h3>
              <div className="flex gap-2">
                <button onClick={() => setSentence('')} className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-red-400 transition">
                  <Trash2 size={18} />
                </button>
                <button onClick={() => speak(sentence)} className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-green-400 transition">
                  <Volume2 size={18} />
                </button>
              </div>
            </div>
            <p className="text-3xl font-medium min-h-[4rem] break-words">
              {sentence || <span className="text-slate-600 italic">Start signing to form a sentence...</span>}
            </p>
          </div>

          {/* Translation Output */}
          <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-4">
                <h3 className="text-slate-400 font-bold uppercase text-xs tracking-widest">Translation</h3>
                <select 
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="bg-slate-700 border-none rounded-md text-xs py-1 px-2 focus:ring-2 focus:ring-blue-500"
                >
                  <option value="hi">Hindi</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
              <div className="flex gap-2">
                <button onClick={translate} className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-blue-400 transition">
                  <Languages size={18} />
                </button>
                <button onClick={() => speak(translatedText)} className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-green-400 transition">
                  <Volume2 size={18} />
                </button>
              </div>
            </div>
            <p className="text-2xl font-medium text-blue-200 min-h-[3rem] break-words">
              {translatedText || <span className="text-slate-600 italic">Click translate to see results...</span>}
            </p>
          </div>

          {/* Tips Section */}
          <div className="bg-blue-600/10 rounded-2xl p-6 border border-blue-500/30">
            <h4 className="flex items-center gap-2 text-blue-400 font-bold text-sm mb-3">
              <Settings size={16} />
              Quick Controls
            </h4>
            <div className="grid grid-cols-2 gap-4 text-xs text-blue-200/70">
              <div className="flex items-center gap-2">
                <kbd className="bg-slate-800 px-2 py-1 rounded border border-slate-600 text-white font-mono">Space</kbd>
                <span>Capture Character</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="bg-slate-800 px-2 py-1 rounded border border-slate-600 text-white font-mono">Enter</kbd>
                <span>Add Space</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="bg-slate-800 px-2 py-1 rounded border border-slate-600 text-white font-mono">Backspace</kbd>
                <span>Delete Last Char</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="bg-slate-800 px-2 py-1 rounded border border-slate-600 text-white font-mono">Q</kbd>
                <span>Toggle Mode</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Teaching Module Overlay */}
      {isTeaching && (
        <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-lg z-50 flex flex-col p-12">
          <div className="flex justify-between items-center mb-12">
            <div>
              <h2 className="text-4xl font-black mb-2">Teaching Module</h2>
              <p className="text-slate-400">Master {mode} gestures through real-time feedback</p>
            </div>
            <button 
              onClick={() => setIsTeaching(false)}
              className="bg-slate-800 hover:bg-red-900/40 p-3 rounded-full transition"
            >
              <Trash2 size={24} />
            </button>
          </div>

          <div className="flex-1 grid grid-cols-3 gap-12">
            <div className="col-span-1 flex flex-col items-center justify-center bg-slate-900 rounded-3xl border border-slate-800 p-8 shadow-inner">
              <span className="text-slate-500 font-bold uppercase tracking-widest text-sm mb-4">Target Gesture</span>
              <div className="text-[12rem] font-black leading-none text-blue-500 drop-shadow-[0_0_30px_rgba(59,130,246,0.5)]">
                {targetLetter}
              </div>
              <div className="mt-8 flex gap-2 overflow-x-auto p-4 w-full justify-center">
                {(mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET).slice(0, 10).map(char => (
                  <button 
                    key={char}
                    onClick={() => setTargetLetter(char)}
                    className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold transition ${targetLetter === char ? 'bg-blue-600 scale-110 shadow-lg' : 'bg-slate-800 hover:bg-slate-700'}`}
                  >
                    {char}
                  </button>
                ))}
              </div>
            </div>

            <div className="col-span-2 relative rounded-3xl overflow-hidden border-8 border-slate-900 shadow-2xl bg-black">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full h-full object-cover grayscale opacity-50"
              />
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                {prediction === targetLetter ? (
                  <div className="bg-green-500/20 border-4 border-green-500 backdrop-blur-md p-12 rounded-full animate-bounce">
                    <span className="text-8xl">✅</span>
                  </div>
                ) : (
                  <div className="bg-white/5 border-2 border-white/20 backdrop-blur-md p-8 rounded-2xl flex flex-col items-center">
                    <span className="text-slate-400 font-medium mb-2">You are signing</span>
                    <span className="text-6xl font-black text-white">{prediction || '?'}</span>
                  </div>
                )}
                
                <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-md">
                   <div className="h-4 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                      <div 
                        className="h-full bg-green-500 transition-all duration-300" 
                        style={{ width: `${prediction === targetLetter ? confidence * 100 : 0}%` }} 
                      />
                   </div>
                   <p className="text-center mt-3 font-bold text-slate-400">Match Accuracy</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
