import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Message {
  id: string;
  sender: 'user' | 'system';
  text: string;
  engineMode?: 'local' | 'cloud';
  citations?: string[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      sender: 'system',
      text: 'Mabuhay! Welcome to OmniJuris. I am grounded directly in Philippine Supreme Court jurisprudence, republic acts, and executive issuances. How can I assist your legal research today?'    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [llmMode, setLlmMode] = useState<'local' | 'cloud'>('local');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');

    setMessages(prev => [...prev, { id: crypto.randomUUID(), sender: 'user', text: userQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, override_mode: llmMode }),
      });

      if (!response.ok) throw new Error('Network error');
      const data = await response.json();

      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        sender: 'system',
        text: data.answer,
        engineMode: data.engine_mode,
        citations: data.citations || []
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        sender: 'system',
        text: 'Error connecting to OmniJuris backend. Ensure your FastAPI server is running.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="omnijuris-workspace">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <span className="ai-avatar">
          <img src="/omnijuris.png" alt="OmniJuris Logo" className="side-avatar-img" />
        </span>
        <div className="sidebar-header">
          <h2>OmniJuris</h2>
          <p>Philippine Legal Intelligence</p>
        </div>

        <div className="config-group">
          <label className="config-label">Inference Engine</label>
          {/* Stacked Vertical Controls */}
          <div className="vertical-controls">
            <button 
              type="button" 
              className={`engine-card ${llmMode === 'local' ? 'active local-active' : ''}`} 
              onClick={() => setLlmMode('local')}
            >
              <div className="engine-card-header">
                <span className="engine-title">Local Engine</span>
                <span className="engine-status-tag">Air-Gapped</span>
              </div>
              <p className="engine-details">Gemma2 27B via Ollama. Completely offline.</p>
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
              <p className="engine-details">Gemini Production API. High throughput.</p>
            </button>
          </div>
        </div>

        <div className="sidebar-footer">
          <div>Dataset: OmniCorpus (2024)</div>
        </div>
      </aside>

      {/* Main Framework Viewport */}
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
                      {msg.engineMode && <span className="badge">{msg.engineMode}</span>}
                    </div>
                    <div className="body-text">{msg.text}</div>
                    
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

        {/* Input Interface */}
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