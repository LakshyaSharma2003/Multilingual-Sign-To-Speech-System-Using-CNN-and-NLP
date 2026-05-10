import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { 
  Settings, 
  Languages, 
  Trash2, 
  BookOpen, 
  Volume2,
  GraduationCap,
  HelpCircle,
  ListChecks,
  RotateCcw,
  CheckCircle2,
  XCircle
} from 'lucide-react';

const ASL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const ISL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

type Module = 'detect' | 'learn' | 'quiz' | 'help';
type QuizOption = {
  letter: string;
  isCorrect: boolean;
};

const gestureNotes: Record<string, string> = {
  A: 'Closed hand with thumb resting beside the fingers.',
  B: 'Flat upright palm with fingers together.',
  C: 'Curved hand shape, like holding a cup.',
  D: 'Index finger raised while other fingers touch the thumb.',
  E: 'Fingers folded toward the palm.',
  F: 'Index finger and thumb touch, other fingers raised.',
  G: 'Index finger and thumb point sideways.',
  H: 'Index and middle fingers extended together.',
  I: 'Little finger raised with other fingers closed.',
  J: 'Little finger draws a J-shaped movement.',
  K: 'Index and middle fingers raised with thumb between them.',
  L: 'Thumb and index finger form an L shape.',
  M: 'Thumb tucked under three fingers.',
  N: 'Thumb tucked under two fingers.',
  O: 'Fingers and thumb form a rounded O shape.',
  P: 'K-like hand shape angled downward.',
  Q: 'G-like hand shape angled downward.',
  R: 'Index and middle fingers crossed.',
  S: 'Closed fist with thumb across the front.',
  T: 'Thumb tucked between index and middle finger.',
  U: 'Index and middle fingers raised together.',
  V: 'Index and middle fingers raised apart.',
  W: 'Three fingers raised to form W.',
  X: 'Index finger bent like a hook.',
  Y: 'Thumb and little finger extended.',
  Z: 'Index finger draws a Z-shaped movement.'
};

const getLearningCards = (selectedMode: 'ASL' | 'ISL') =>
  (selectedMode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET).map((letter, index) => ({
    letter,
    note: gestureNotes[letter],
    tip: selectedMode === 'ASL'
      ? `Practice ${letter} with one clear hand shape in frame.`
      : `Practice ${letter} with ${index % 3 === 0 ? 'both hands visible when needed' : 'your hand centered and steady'}.`
  }));

const makeQuizOptions = (correct: string, alphabet: string[]): QuizOption[] => {
  const distractors = alphabet.filter(letter => letter !== correct).sort(() => 0.5 - Math.random()).slice(0, 3);
  return [...distractors, correct]
    .sort(() => 0.5 - Math.random())
    .map(letter => ({ letter, isCorrect: letter === correct }));
};

const GestureReference: React.FC<{ letter: string; mode: 'ASL' | 'ISL'; compact?: boolean }> = ({ letter, mode, compact = false }) => {
  const seed = letter.charCodeAt(0) - 64;
  const fingers = Array.from({ length: 5 }, (_, index) => {
    const active = ((seed + index) % 3) !== 0;
    const height = active ? 42 + ((seed + index * 7) % 28) : 20 + ((seed + index * 5) % 12);
    return { active, height, x: 24 + index * 19 };
  });

  return (
    <svg
      viewBox="0 0 132 132"
      role="img"
      aria-label={`${mode} ${letter} gesture reference`}
      className={`${compact ? 'h-24' : 'h-32'} w-full rounded-lg bg-slate-950/70`}
    >
      <rect x="1" y="1" width="130" height="130" rx="8" fill="#020617" stroke="#334155" />
      {fingers.map((finger, index) => (
        <rect
          key={index}
          x={finger.x}
          y={72 - finger.height}
          width="13"
          height={finger.height}
          rx="6"
          fill={finger.active ? '#60a5fa' : '#475569'}
        />
      ))}
      <rect x="28" y="66" width="72" height="42" rx="18" fill="#93c5fd" />
      <circle cx={83} cy={82} r={11} fill={mode === 'ASL' ? '#facc15' : '#34d399'} />
      <text x="66" y="121" textAnchor="middle" fill="#e2e8f0" fontSize="16" fontWeight="700">
        {mode} {letter}
      </text>
    </svg>
  );
};

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
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [activeModule, setActiveModule] = useState<Module>('detect');
  const [quizQuestion, setQuizQuestion] = useState<number>(0);
  const [quizScore, setQuizScore] = useState<number>(0);
  const [quizFeedback, setQuizFeedback] = useState<'correct' | 'wrong' | null>(null);
  const [teachingStreak, setTeachingStreak] = useState<number>(0);
  const [bestTeachingStreak, setBestTeachingStreak] = useState<number>(0);
  const [teachingAttempts, setTeachingAttempts] = useState<number>(0);
  
  const webcamRef = useRef<Webcam>(null);
  const ws = useRef<WebSocket | null>(null);
  const awaitingPrediction = useRef<boolean>(false);
  const sentFrameCount = useRef<number>(0);
  const predictionTimeout = useRef<number | null>(null);
  const lastTeachingHit = useRef<string>('');
  const learningCards = getLearningCards(mode);
  const quizTarget = learningCards[quizQuestion % learningCards.length];
  const [quizOptions, setQuizOptions] = useState<QuizOption[]>(() => makeQuizOptions('A', ASL_ALPHABET));

  // Initialize WebSocket
  useEffect(() => {
    const connect = () => {
      setWsStatus('connecting');
      ws.current = new WebSocket('ws://localhost:8000/ws/predict');
      
      ws.current.onopen = () => setWsStatus('connected');
      ws.current.onerror = () => {
        awaitingPrediction.current = false;
        if (predictionTimeout.current) {
          window.clearTimeout(predictionTimeout.current);
        }
        setWsStatus('error');
      };
      ws.current.onclose = () => {
        awaitingPrediction.current = false;
        if (predictionTimeout.current) {
          window.clearTimeout(predictionTimeout.current);
        }
        setWsStatus('connecting');
        setTimeout(connect, 3000); // Reconnect after 3s
      };

      ws.current.onmessage = (event) => {
        awaitingPrediction.current = false;
        if (predictionTimeout.current) {
          window.clearTimeout(predictionTimeout.current);
          predictionTimeout.current = null;
        }
        const data = JSON.parse(event.data);
        setPrediction(data.letter || '');
        setConfidence(data.confidence || 0);
        if (data.image) {
          setProcessedImage(data.image);
        }
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, []);

  // Frame Capture and Sending
  const capture = useCallback(() => {
    const shouldStream = activeModule === 'detect' || isTeaching;

    if (shouldStream && webcamRef.current && ws.current?.readyState === WebSocket.OPEN && !awaitingPrediction.current) {
      const imageSrc = webcamRef.current.getScreenshot({ width: 424, height: 318 });
      if (imageSrc) {
        sentFrameCount.current += 1;
        awaitingPrediction.current = true;
        ws.current.send(JSON.stringify({
          image: imageSrc,
          mode: mode,
          include_image: sentFrameCount.current % 3 === 0
        }));
        predictionTimeout.current = window.setTimeout(() => {
          awaitingPrediction.current = false;
          predictionTimeout.current = null;
        }, 1200);
      }
    }
  }, [activeModule, isTeaching, mode]);

  useEffect(() => {
    const interval = setInterval(capture, 120);
    return () => clearInterval(interval);
  }, [capture]);

  useEffect(() => {
    setQuizQuestion(0);
    setQuizScore(0);
    setQuizFeedback(null);
    setQuizOptions(makeQuizOptions('A', mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET));
    setTargetLetter('A');
    setTeachingStreak(0);
    setTeachingAttempts(0);
    lastTeachingHit.current = '';
  }, [mode]);

  useEffect(() => {
    lastTeachingHit.current = '';
    setTeachingStreak(0);
  }, [targetLetter]);

  useEffect(() => {
    const isMatch = isTeaching && prediction === targetLetter && confidence >= 0.65;

    if (isMatch && lastTeachingHit.current !== targetLetter) {
      lastTeachingHit.current = targetLetter;
      setTeachingAttempts(prev => prev + 1);
      setTeachingStreak(prev => {
        const next = prev + 1;
        setBestTeachingStreak(best => Math.max(best, next));
        return next;
      });
    } else if (!isMatch && prediction && prediction !== targetLetter) {
      lastTeachingHit.current = '';
      setTeachingStreak(0);
    }
  }, [confidence, isTeaching, prediction, targetLetter]);

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
      } else if (e.code === 'KeyQ') {
        setMode(prev => prev === 'ASL' ? 'ISL' : 'ASL');
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

  const answerQuiz = (option: QuizOption) => {
    if (quizFeedback) return;

    if (option.isCorrect) {
      setQuizScore(prev => prev + 1);
      setQuizFeedback('correct');
    } else {
      setQuizFeedback('wrong');
    }
  };

  const nextQuizQuestion = () => {
    const nextIndex = (quizQuestion + 1) % learningCards.length;
    setQuizQuestion(nextIndex);
    setQuizFeedback(null);
    setQuizOptions(makeQuizOptions(learningCards[nextIndex].letter, mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET));
  };

  const resetQuiz = () => {
    setQuizQuestion(0);
    setQuizScore(0);
    setQuizFeedback(null);
    setQuizOptions(makeQuizOptions('A', mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET));
  };

  const moduleButtonClass = (module: Module) =>
    `flex items-center gap-2 px-4 py-2 rounded-lg transition ${
      activeModule === module ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
    }`;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      <header className="flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Languages className="text-blue-400" />
          Multilingual Sign-to-Speech
        </h1>
        <div className="flex gap-4">
          <div className="flex gap-2">
            <button onClick={() => setActiveModule('detect')} className={moduleButtonClass('detect')}>
              <Languages size={18} />
              Detect
            </button>
            <button onClick={() => setActiveModule('learn')} className={moduleButtonClass('learn')}>
              <GraduationCap size={18} />
              Learn
            </button>
            <button onClick={() => setActiveModule('quiz')} className={moduleButtonClass('quiz')}>
              <ListChecks size={18} />
              Quiz
            </button>
            <button onClick={() => setActiveModule('help')} className={moduleButtonClass('help')}>
              <HelpCircle size={18} />
              Help
            </button>
          </div>
          <button 
            onClick={() => setIsTeaching(!isTeaching)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${isTeaching ? 'bg-blue-600' : 'bg-slate-700 hover:bg-slate-600'}`}
          >
            <BookOpen size={20} />
            {mode} Teaching Module
          </button>
          <div className="flex bg-slate-800 rounded-lg p-1">
            <button 
              onClick={() => {
                setMode('ASL');
                setPrediction('');
                setConfidence(0);
              }}
              className={`px-4 py-1 rounded-md transition ${mode === 'ASL' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
            >
              ASL
            </button>
            <button 
              onClick={() => {
                setMode('ISL');
                setPrediction('');
                setConfidence(0);
              }}
              className={`px-4 py-1 rounded-md transition ${mode === 'ISL' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
            >
              ISL
            </button>
          </div>
        </div>
      </header>

      {activeModule === 'detect' && (
      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Camera Section */}
        <div className="relative rounded-2xl overflow-hidden bg-black aspect-video border-4 border-slate-800 shadow-2xl">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.55}
            className="w-full h-full object-cover"
            videoConstraints={{ 
              width: 424,
              height: 318,
              facingMode: "user" 
            }}
          />

          {processedImage && (
            <img 
              src={processedImage} 
              alt="Processed Feed" 
              className="absolute inset-0 w-full h-full object-cover"
            />
          )}

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
      )}

      {activeModule === 'learn' && (
        <main className="space-y-6">
          <section className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
              <div>
                <h2 className="text-3xl font-black">{mode} Alphabet Learning Module</h2>
                <p className="text-slate-400 mt-2 max-w-3xl">
                  Study each alphabet sign, review the visual reference, then switch to Teaching Module or Quiz to practice recognition.
                </p>
              </div>
              <button onClick={() => setIsTeaching(true)} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 transition">
                <BookOpen size={18} />
                Practice Selected Mode
              </button>
            </div>
          </section>

          <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {learningCards.map(card => (
              <article key={card.letter} className="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow-lg">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl font-black text-blue-300">{card.letter}</span>
                  <button
                    onClick={() => {
                      setTargetLetter(card.letter);
                      setIsTeaching(true);
                    }}
                    className="px-3 py-1 rounded-md bg-slate-700 hover:bg-slate-600 text-sm transition"
                  >
                    Practice
                  </button>
                </div>
                <GestureReference letter={card.letter} mode={mode} />
                <p className="mt-3 text-sm text-slate-200">{card.note}</p>
                <p className="mt-2 text-xs text-slate-400">{card.tip}</p>
              </article>
            ))}
          </section>
        </main>
      )}

      {activeModule === 'quiz' && (
        <main className="grid grid-cols-1 lg:grid-cols-[0.9fr_1.1fr] gap-8">
          <section className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-3xl font-black">{mode} Quiz</h2>
                <p className="text-slate-400 mt-1">Question {quizQuestion + 1} of {learningCards.length}</p>
              </div>
              <button onClick={resetQuiz} className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition">
                <RotateCcw size={20} />
              </button>
            </div>

            <GestureReference letter={quizTarget.letter} mode={mode} />
            <p className="mt-5 text-slate-300">Which alphabet does this reference card represent?</p>

            <div className="grid grid-cols-2 gap-3 mt-5">
              {quizOptions.map(option => {
                const showResult = quizFeedback && option.isCorrect;
                const showWrong = quizFeedback === 'wrong' && !option.isCorrect;
                return (
                  <button
                    key={option.letter}
                    onClick={() => answerQuiz(option)}
                    className={`h-16 rounded-xl border text-2xl font-black transition ${
                      showResult
                        ? 'bg-green-500/20 border-green-400 text-green-200'
                        : showWrong
                          ? 'bg-red-500/10 border-red-500/40 text-red-200'
                          : 'bg-slate-900 border-slate-700 hover:border-blue-400'
                    }`}
                  >
                    {option.letter}
                  </button>
                );
              })}
            </div>

            {quizFeedback && (
              <div className="mt-5 flex items-center justify-between gap-4 bg-slate-900 border border-slate-700 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  {quizFeedback === 'correct' ? <CheckCircle2 className="text-green-400" /> : <XCircle className="text-red-400" />}
                  <span className="font-bold">
                    {quizFeedback === 'correct' ? 'Correct answer' : `Correct answer: ${quizTarget.letter}`}
                  </span>
                </div>
                <button onClick={nextQuizQuestion} className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 transition">
                  Next
                </button>
              </div>
            )}
          </section>

          <section className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
            <h3 className="text-slate-400 font-bold uppercase text-xs tracking-widest mb-4">Quiz Progress</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900 rounded-xl p-5 border border-slate-700">
                <p className="text-slate-500 text-sm">Score</p>
                <p className="text-5xl font-black text-blue-300 mt-2">{quizScore}</p>
              </div>
              <div className="bg-slate-900 rounded-xl p-5 border border-slate-700">
                <p className="text-slate-500 text-sm">Mode</p>
                <p className="text-5xl font-black text-blue-300 mt-2">{mode}</p>
              </div>
            </div>
            <div className="mt-6 space-y-3">
              <p className="text-slate-300">Use the quiz after reviewing the learning module. The visual cards are generated references, and the live model feedback is available in Teaching Module.</p>
              <button onClick={() => setActiveModule('learn')} className="w-full px-4 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 transition">
                Review Alphabet Module
              </button>
            </div>
          </section>
        </main>
      )}

      {activeModule === 'help' && (
        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {[
            {
              title: 'Detection',
              items: ['Keep your hand centered in the webcam frame.', 'Use Space to capture the current prediction.', 'Use Enter to add a space and Backspace to delete.', 'Use Q or the ASL/ISL buttons to switch modes.']
            },
            {
              title: 'Learning',
              items: ['Open Learn to review all alphabet references.', 'Use Practice on any card to open live feedback.', 'Match your gesture with the target letter in Teaching Module.', 'Use steady lighting for clearer landmarks.']
            },
            {
              title: 'Translation & Speech',
              items: ['Build a sentence from detected letters.', 'Choose Hindi, Spanish, French, or German.', 'Use the translate icon to translate the sentence.', 'Use the speaker icon to hear original or translated text.']
            }
          ].map(section => (
            <section key={section.title} className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg">
              <h2 className="text-2xl font-black mb-4">{section.title}</h2>
              <div className="space-y-3">
                {section.items.map(item => (
                  <div key={item} className="flex gap-3 text-slate-300">
                    <CheckCircle2 size={18} className="text-blue-400 mt-1 shrink-0" />
                    <p>{item}</p>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </main>
      )}

      {/* Teaching Module Overlay */}
      {isTeaching && (
        <div className="fixed inset-0 bg-slate-950/95 backdrop-blur-lg z-50 flex flex-col p-8 overflow-y-auto">
          <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-4 mb-8">
            <div>
              <p className="text-blue-300 font-bold uppercase tracking-widest text-xs mb-2">Live practice arena</p>
              <h2 className="text-4xl font-black mb-2">{mode} Teaching Module</h2>
              <p className="text-slate-400">Match the target, hold your hand steady, and build a streak with real-time model feedback.</p>
            </div>
            <div className="grid grid-cols-3 gap-3 min-w-[360px]">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <p className="text-slate-500 text-xs uppercase font-bold">Streak</p>
                <p className="text-3xl font-black text-green-300">{teachingStreak}</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <p className="text-slate-500 text-xs uppercase font-bold">Best</p>
                <p className="text-3xl font-black text-blue-300">{bestTeachingStreak}</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <p className="text-slate-500 text-xs uppercase font-bold">Hits</p>
                <p className="text-3xl font-black text-yellow-300">{teachingAttempts}</p>
              </div>
            </div>
            <button 
              onClick={() => setIsTeaching(false)}
              className="bg-slate-800 hover:bg-red-900/40 p-3 rounded-full transition"
            >
              <Trash2 size={24} />
            </button>
          </div>

          <div className="flex-1 grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-8 min-h-[640px]">
            <aside className="flex flex-col gap-5">
              <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-inner">
                <span className="text-slate-500 font-bold uppercase tracking-widest text-sm">Target Gesture</span>
                <div className="mt-4 grid grid-cols-[1fr_120px] gap-4 items-center">
                  <div className="text-[9rem] font-black leading-none text-blue-500 drop-shadow-[0_0_30px_rgba(59,130,246,0.45)]">
                    {targetLetter}
                  </div>
                  <GestureReference letter={targetLetter} mode={mode} compact />
                </div>
                <p className="text-sm text-slate-300 mt-4">{gestureNotes[targetLetter]}</p>
                <div className="mt-5 h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                  <div
                    className={`h-full transition-all duration-300 ${prediction === targetLetter ? 'bg-green-400' : 'bg-blue-500'}`}
                    style={{ width: `${prediction === targetLetter ? confidence * 100 : Math.min(confidence * 45, 45)}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  Current confidence: {(confidence * 100).toFixed(1)}%
                </p>
              </div>

              <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-slate-300 font-bold">Choose Letter</h3>
                  <button
                    onClick={() => {
                      const alphabet = mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET;
                      const current = alphabet.indexOf(targetLetter);
                      setTargetLetter(alphabet[(current + 1) % alphabet.length]);
                    }}
                    className="px-3 py-1 rounded-md bg-blue-600 hover:bg-blue-500 text-sm transition"
                  >
                    Next
                  </button>
                </div>
                <div className="grid grid-cols-6 gap-2">
                {(mode === 'ASL' ? ASL_ALPHABET : ISL_ALPHABET).map(char => (
                  <button 
                    key={char}
                    onClick={() => setTargetLetter(char)}
                    className={`h-10 rounded-lg flex items-center justify-center font-bold transition ${
                      targetLetter === char ? 'bg-blue-600 scale-105 shadow-lg' : 'bg-slate-800 hover:bg-slate-700'
                    }`}
                  >
                    {char}
                  </button>
                ))}
                </div>
              </div>

              <div className="bg-blue-600/10 rounded-2xl border border-blue-500/30 p-5">
                <h3 className="font-bold text-blue-300 mb-3">Practice Loop</h3>
                <div className="space-y-2 text-sm text-blue-100/80">
                  <p>1. Pick a target letter.</p>
                  <p>2. Hold the gesture until the match turns green.</p>
                  <p>3. Move to the next letter and keep the streak alive.</p>
                </div>
              </div>
            </aside>

            <section className="relative rounded-3xl overflow-hidden border-8 border-slate-900 shadow-2xl bg-black min-h-[620px]">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.55}
                className="w-full h-full object-cover grayscale opacity-50"
                videoConstraints={{
                  width: 424,
                  height: 318,
                  facingMode: "user"
                }}
              />
              
              {processedImage && (
                <img 
                  src={processedImage} 
                  alt="Processed Feed" 
                  className="absolute inset-0 w-full h-full object-cover grayscale opacity-50"
                />
              )}

              <div className="absolute inset-0 flex flex-col items-center justify-center">
                {prediction === targetLetter ? (
                  <div className="bg-green-500/20 border-4 border-green-500 backdrop-blur-md px-12 py-8 rounded-3xl shadow-[0_0_40px_rgba(34,197,94,0.35)]">
                    <span className="text-6xl font-black text-green-100">Correct</span>
                    <p className="text-center text-green-200 mt-2">Hold steady, then try the next letter.</p>
                  </div>
                ) : (
                  <div className="bg-white/5 border-2 border-white/20 backdrop-blur-md p-8 rounded-2xl flex flex-col items-center min-w-[240px]">
                    <span className="text-slate-400 font-medium mb-2">Detected</span>
                    <span className="text-7xl font-black text-white">{prediction || '?'}</span>
                    <span className="text-sm text-slate-400 mt-2">Target: {targetLetter}</span>
                  </div>
                )}
                
                <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-lg px-4">
                   <div className="h-4 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                      <div 
                        className={`h-full transition-all duration-300 ${prediction === targetLetter ? 'bg-green-500' : 'bg-blue-500'}`}
                        style={{ width: `${prediction === targetLetter ? confidence * 100 : 0}%` }} 
                      />
                   </div>
                   <p className="text-center mt-3 font-bold text-slate-300">
                    {prediction === targetLetter ? 'Match Accuracy' : 'Waiting for target match'}
                   </p>
                </div>
              </div>
            </section>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
