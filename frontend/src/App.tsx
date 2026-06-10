import { useState, useRef, useEffect } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  sender: 'user' | 'system';
  text: string;
  engineMode?: 'local' | 'cloud';
  citations?: string[];
}

function App() {
  const isHosted = import.meta.env.VITE_IS_HOSTED === 'true';

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      sender: 'system',
      text: 'Mabuhay! Welcome to OmniJuris. I am grounded directly in Philippine Supreme Court jurisprudence, republic acts, and executive issuances. How can I assist your legal research today?'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const [llmMode, setLlmMode] = useState<'local' | 'cloud'>(isHosted ? 'cloud' : 'local');
  const [thinkingMode, setThinkingMode] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (llmMode === 'cloud') {
      setThinkingMode(false);
    }
  }, [llmMode]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');

    const userMsg: Message = { id: crypto.randomUUID(), sender: 'user', text: userQuery };
    const assistantMsg: Message = { id: crypto.randomUUID(), sender: 'system', text: '', engineMode: llmMode };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsLoading(true);

    const backendUrl = isHosted ? 'https://your-render-backend-url.com/query/stream' : 'http://localhost:8000/query/stream';

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          override_mode: llmMode,
          thinking_mode: thinkingMode
        }),
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let citations: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.done) {
              citations = data.citations || [];
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMsg.id
                  ? { ...msg, citations }
                  : msg
              ));
            } else if (data.token) {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMsg.id
                  ? { ...msg, text: msg.text + data.token }
                  : msg
              ));
            }
          } catch {}
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsg.id
          ? { ...msg, text: 'Error connecting to OmniJuris backend.' }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="omnijuris-workspace">
      {/* Sidebar */}
      <aside className="sidebar">
        <span className="ai-avatar">
          <img src="/omnijuris.png" alt="OmniJuris Logo" className="side-avatar-img" />
        </span>
        <div className="sidebar-header">
          <h2>OmniJuris</h2>
          <p>Philippine Legal Intelligence</p>
        </div>

        {/* Inference Engine */}
        <div className="config-group">
          <label className="config-label">Inference Engine</label>
          <div className="vertical-controls">
            <button
              type="button"
              className={`engine-card ${llmMode === 'local' ? 'active local-active' : ''} ${isHosted ? 'disabled-card' : ''}`}
              onClick={() => setLlmMode('local')}
              disabled={isHosted} // 3. Disable the button entirely if hosted
            >
              <div className="engine-card-header">
                <span className="engine-title">Local Engine</span>
                <span className="engine-status-tag">{isHosted ? 'UNAVAILABLE' : 'Air-Gapped'}</span>
              </div>
              <p className="engine-details">
                {isHosted 
                  ? 'Local Ollama node is disabled in the web demo.' 
                  : 'Qwen3 4B via Ollama. Completely offline.'}
              </p>
            </button>

            <button
              type="button"
              className={`engine-card ${llmMode === 'cloud' ? 'active cloud-active' : ''}`}
              onClick={() => setLlmMode('cloud')}
            >
              <div className="engine-card-header">
                <span className="engine-title">Cloud Engine</span>
                <span className="engine-status-tag">API</span>
              </div>
              <p className="engine-details">Gemini 3.5 Flash. High throughput.</p>
            </button>
          </div>
        </div>

        {/* Thinking Mode */}
        <div className="config-group">
          <label className="config-label">Reasoning Mode</label>
          <button
            type="button"
            // 4. Bug Fix: Only apply active class if thinking mode is true AND we are actually on local mode
            className={`engine-card ${thinkingMode && llmMode === 'local' ? 'active local-active' : ''}`}
            onClick={() => setThinkingMode(prev => !prev)}
            disabled={llmMode === 'cloud'}
          >
            <div className="engine-card-header">
              <span className="engine-title">Extended Thinking</span>
              <span className="engine-status-tag">{thinkingMode && llmMode === 'local' ? 'ON' : 'OFF'}</span>
            </div>
            <p className="engine-details">
              {llmMode === 'cloud'
                ? 'Not available for cloud engine.'
                : 'Qwen3 reasons step-by-step before answering. Slower but more accurate.'}
            </p>
          </button>
        </div>

        <div className="sidebar-footer">
          <div>Dataset: OmniCorpus (2024)</div>
        </div>
      </aside>

      {/* Main canvas */}
      <main className="main-canvas">
        <div className="viewport-scroll">
          <div className="conversation-flow">
            {messages.map((msg) => (
              <div key={msg.id} className={`chat-row ${msg.sender}`}>
                {msg.sender === 'system' ? (
                  <div className="ai-content-canvas">
                    <div className="author-meta">
                      <span className="ai-avatar">
                        <img src="/omnijuris.png" alt="OmniJuris Logo" className="avatar-img" />
                      </span>
                      <span className="author-name">OmniJuris</span>
                      {msg.engineMode && (
                        <span className="badge">
                          {msg.engineMode === 'local' ? 'Qwen3 4B' : 'Gemini 3.5 Flash'}
                        </span>
                      )}
                      {msg.engineMode === 'local' && thinkingMode && (
                        <span className="badge">thinking</span>
                      )}
                    </div>
                    <div className="body-text">
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </div>

                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-tray">
                        {msg.citations.map((cite, idx) => (
                          <span key={idx} className="cite-pill">🏛️ {cite}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="user-bubble-wrapper">
                    <div className="user-bubble">
                      <div className="body-text">{msg.text}</div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="chat-row system">
                <div className="ai-content-canvas clean-loading">
                  <div className="author-meta">
                    <span className="ai-avatar processing-pulse">
                      <img src="/omnijuris.png" alt="OmniJuris Logo" className="avatar-img" />
                    </span>
                    <span className="author-name">OmniJuris</span>
                  </div>
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <footer className="input-zone">
          <form onSubmit={handleSendMessage} className="floating-bar">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Supreme Court precedents, rules, or statutes..."
              disabled={isLoading}
            />
            <button type="submit" className="send-btn" disabled={isLoading || !input.trim()}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

export default App;