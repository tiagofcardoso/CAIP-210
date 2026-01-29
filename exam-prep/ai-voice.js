// AI Voice Services - Speech Recognition and Synthesis
// Handles voice input (Speech-to-Text) and voice output (Text-to-Speech)

class VoiceInput {
    constructor() {
        // Check browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            console.warn('Speech Recognition not supported in this browser');
            this.supported = false;
            return;
        }

        this.supported = true;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.isRecording = false;

        // Set language based on current app language
        this.updateLanguage();
    }

    updateLanguage() {
        if (!this.supported) return;
        const lang = (typeof currentLanguage !== 'undefined' ? currentLanguage : 'pt');
        this.recognition.lang = lang === 'pt' ? 'pt-BR' : 'en-US';
        console.log('[VoiceInput] Language set to:', this.recognition.lang);
    }

    start(onInterim, onFinal) {
        if (!this.supported || this.isRecording) return Promise.reject('Not supported or already recording');

        return new Promise((resolve, reject) => {
            this.isRecording = true;

            this.recognition.onresult = (event) => {
                const results = Array.from(event.results);
                const transcript = results
                    .map(result => result[0].transcript)
                    .join('');

                // Call interim callback for real-time feedback
                if (onInterim && !event.results[event.results.length - 1].isFinal) {
                    onInterim(transcript);
                }

                // Final result
                if (event.results[event.results.length - 1].isFinal) {
                    if (onFinal) onFinal(transcript);
                    resolve(transcript);
                    this.isRecording = false;
                }
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.isRecording = false;
                reject(event.error);
            };

            this.recognition.onend = () => {
                this.isRecording = false;
            };

            try {
                this.recognition.start();
            } catch (error) {
                this.isRecording = false;
                reject(error);
            }
        });
    }

    stop() {
        if (!this.supported || !this.isRecording) return;
        this.recognition.stop();
        this.isRecording = false;
    }
}

class VoiceOutput {
    constructor() {
        this.synth = window.speechSynthesis;
        this.supported = 'speechSynthesis' in window;
        this.autoPlay = localStorage.getItem('ai_auto_speak') === 'true';
        this.isSpeaking = false;
        this.voices = [];

        if (this.supported) {
            // Load voices
            this.loadVoices();

            // Voices might load asynchronously
            if (speechSynthesis.onvoiceschanged !== undefined) {
                speechSynthesis.onvoiceschanged = () => this.loadVoices();
            }
        }
    }

    loadVoices() {
        this.voices = this.synth.getVoices();
    }

    getVoiceForLanguage(lang) {
        const langCode = lang.split('-')[0]; // 'pt' or 'en'

        // Prefer Google voices, then Microsoft, then any
        const googleVoice = this.voices.find(v =>
            v.lang.startsWith(langCode) && v.name.includes('Google')
        );

        if (googleVoice) return googleVoice;

        const microsoftVoice = this.voices.find(v =>
            v.lang.startsWith(langCode) && v.name.includes('Microsoft')
        );

        if (microsoftVoice) return microsoftVoice;

        // Fallback to any voice for the language
        return this.voices.find(v => v.lang.startsWith(langCode));
    }

    speak(text, language = null, onStart = null, onEnd = null) {
        if (!this.supported) return Promise.reject('Speech synthesis not supported');

        // Cancel any ongoing speech
        this.synth.cancel();

        return new Promise((resolve, reject) => {
            const utterance = new SpeechSynthesisUtterance(text);

            // Set language
            const currentLang = (typeof currentLanguage !== 'undefined' ? currentLanguage : 'pt');
            const lang = language || (currentLang === 'pt' ? 'pt-BR' : 'en-US');
            utterance.lang = lang;
            console.log('[VoiceOutput] Speaking in language:', lang);

            // Set voice
            const voice = this.getVoiceForLanguage(lang);
            if (voice) utterance.voice = voice;

            // Set speech parameters
            utterance.rate = 1.0;  // Normal speed
            utterance.pitch = 1.0; // Normal pitch
            utterance.volume = 1.0; // Full volume

            // Event handlers
            utterance.onstart = () => {
                this.isSpeaking = true;
                if (onStart) onStart();
            };

            utterance.onend = () => {
                this.isSpeaking = false;
                if (onEnd) onEnd();
                resolve();
            };

            utterance.onerror = (event) => {
                this.isSpeaking = false;
                console.error('Speech synthesis error:', event.error);
                reject(event.error);
            };

            // Speak
            this.synth.speak(utterance);
        });
    }

    stop() {
        if (!this.supported) return;
        this.synth.cancel();
        this.isSpeaking = false;
    }

    toggleAutoPlay() {
        this.autoPlay = !this.autoPlay;
        localStorage.setItem('ai_auto_speak', this.autoPlay);
        return this.autoPlay;
    }

    getAutoPlay() {
        return this.autoPlay;
    }
}

// Check voice support
function checkVoiceSupport() {
    return {
        speechRecognition: 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,
        speechSynthesis: 'speechSynthesis' in window
    };
}

// Initialize voice services (will be called after DOM ready)
let voiceInput = null;
let voiceOutput = null;

// Safe initialization function
function initVoiceServices() {
    console.log('[Voice] Initializing voice services...');
    console.log('[Voice] Current language:', typeof currentLanguage !== 'undefined' ? currentLanguage : 'undefined');

    voiceInput = new VoiceInput();
    voiceOutput = new VoiceOutput();

    console.log('[Voice] Voice input supported:', voiceInput.supported);
    console.log('[Voice] Voice output supported:', voiceOutput.supported);

    // Expose globally
    window.voiceInput = voiceInput;
    window.voiceOutput = voiceOutput;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVoiceServices);
} else {
    initVoiceServices();
}

// Expose check function globally
window.checkVoiceSupport = checkVoiceSupport;
