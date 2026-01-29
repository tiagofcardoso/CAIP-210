# 🤖 AI Study Assistant - Configuration Guide

## 🔑 Getting Your Gemini API Key

1. **Go to Google AI Studio**: https://makersuite.google.com/app/apikey
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy** the generated API key

## ⚙️ Configuration Steps

### 1. Add Your API Key

Open `exam-prep/ai-assistant.js` and replace the placeholder:

```javascript
// Line 6
const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'; // TODO: Replace this
```

**Replace with:**
```javascript
const GEMINI_API_KEY = 'your-actual-api-key-here';
```

### 2. Test Locally

```bash
cd exam-prep
python -m http.server 8000
```

Open: http://localhost:8000

### 3. Test the AI Assistant

1. Answer a question **incorrectly**
2. Click the **"🤖 Perguntar à IA"** button
3. Try both:
   - **Typing** a question
   - **Speaking** (hold the 🎤 button)
4. Check if AI responds correctly

## 🎤 Voice Features

### Speech-to-Text (Voice Input)
- **Hold** the 🎤 button and speak
- Text appears in real-time
- **Release** to send

### Text-to-Speech (Voice Output)
- Toggle **🔊 Auto-speak** in header
- Click 🔊 on any AI message to replay

## 🌐 Browser Support

| Feature | Chrome/Edge | Safari | Firefox |
|---------|-------------|--------|---------|
| Chat | ✅ Full | ✅ Full | ✅ Full |
| Voice Input | ✅ Full | ✅ iOS 14.5+ | ⚠️ Limited |
| Voice Output | ✅ Full | ✅ Full | ✅ Full |

## 🚀 Deploy to Firebase

### Option 1: With API Key (Simple)

**⚠️ WARNING**: Your API key will be exposed in the frontend!

```bash
cd exam-prep
firebase deploy --only hosting
```

### Option 2: With Firebase Functions (Secure) ⭐ RECOMMENDED

This keeps your API key secret on the server.

#### Step 1: Initialize Functions

```bash
firebase init functions
# Choose JavaScript
# Install dependencies: Yes
```

#### Step 2: Create Function

Create `functions/index.js`:

```javascript
const functions = require('firebase-functions');
const fetch = require('node-fetch');

exports.askAI = functions.https.onCall(async (data, context) => {
    const { message, context: questionContext } = data;
    
    const GEMINI_API_KEY = functions.config().gemini.key;
    const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';
    
    // Build prompt (same logic as ai-assistant.js)
    const systemPrompt = buildSystemPrompt(questionContext);
    const fullPrompt = `${systemPrompt}\n\nAluno: ${message}\n\nTutor:`;
    
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            contents: [{ parts: [{ text: fullPrompt }] }],
            generationConfig: {
                temperature: 0.7,
                maxOutputTokens: 500
            }
        })
    });
    
    const result = await response.json();
    return result.candidates[0].content.parts[0].text;
});

function buildSystemPrompt(context) {
    // Same logic as in ai-assistant.js
    // ...
}
```

#### Step 3: Set API Key

```bash
firebase functions:config:set gemini.key="YOUR_GEMINI_API_KEY"
```

#### Step 4: Update Frontend

Modify `ai-assistant.js` to call the function instead of direct API:

```javascript
async sendMessage(userMessage) {
    const askAI = firebase.functions().httpsCallable('askAI');
    const result = await askAI({
        message: userMessage,
        context: this.currentContext
    });
    return result.data;
}
```

#### Step 5: Deploy

```bash
firebase deploy
```

## 💰 Cost Estimation

### Gemini API Free Tier
- **60 requests/minute**
- **1500 requests/day**
- **$0/month** (free!)

### If You Exceed Free Tier
- **Gemini 1.5 Flash**: $0.075 / 1M input tokens
- **Expected**: ~$5-10/month for 1000 active users

## 🔒 Security Best Practices

1. **Use Firebase Functions** (Option 2) for production
2. **Set rate limiting** on your Firebase project
3. **Monitor usage** in Google Cloud Console
4. **Rotate API keys** periodically

## 🐛 Troubleshooting

### "Please configure your Gemini API key"
- Check if you replaced `YOUR_GEMINI_API_KEY_HERE`
- Make sure there are no extra spaces
- Verify key is valid at https://makersuite.google.com

### "Rate limit exceeded"
- Wait 1 minute
- Reduce usage
- Consider upgrading to paid tier

### Voice not working
- Check browser support (Chrome/Edge recommended)
- Allow microphone permission
- Check if HTTPS (required for voice on production)

### AI responses are slow
- Normal: 2-5 seconds
- Check your internet connection
- Try Gemini 1.5 Flash (faster than Pro)

## 📊 Monitoring

Check usage at:
- **Gemini API**: https://makersuite.google.com/app/apikey
- **Firebase Console**: https://console.firebase.google.com

## 🎯 Next Steps

1. ✅ Add API key
2. ✅ Test locally
3. ✅ Deploy to Firebase
4. 📊 Monitor usage
5. 🎨 Customize prompts (optional)
6. 🚀 Share with users!

---

**Need help?** Check the [implementation plan](../../.gemini/antigravity/brain/2c122591-ec06-4f65-90d1-70a4ae295b6d/implementation_plan.md) for detailed technical documentation.
