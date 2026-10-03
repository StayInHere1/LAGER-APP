import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

// Allow parsing JSON bodies up to 25MB
app.use(express.json({ limit: '25mb' }));

function getGeminiClient(customApiKey) {
  const apiKey = (customApiKey && customApiKey.trim().length > 10) ? customApiKey.trim() : process.env.GEMINI_API_KEY;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Low-latency text generation with automatic model failover against high-demand / quota errors
async function generateTextFast(ai, prompt) {
  // Try gemini-3.1-flash-lite first for lightning-fast answers, failover to gemini-3.8-flash
  const models = ['gemini-3.1-flash-lite', 'gemini-3.8-flash'];
  let lastError = null;

  for (const model of models) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });
      if (response && response.text) {
        return response.text;
      }
    } catch (err) {
      console.warn(`Model ${model} hit error/demand limit, trying alternative:`, err.message);
      lastError = err;
    }
  }
  throw lastError || new Error('KI-Dienst momentan ausgelastet');
}

// Ultra-fast vision recognition with failover and low-thinking mode
async function generateVisionFast(ai, cleanBase64, prompt, mimeType = 'image/jpeg') {
  // gemini-3.8-flash first for accurate vision, gemini-3.1-flash-lite fallback
  const models = ['gemini-3.8-flash', 'gemini-3.1-flash-lite'];
  let lastError = null;

  for (const model of models) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: cleanBase64,
              },
            },
            {
              text: prompt,
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });
      if (response && response.text) {
        return response.text;
      }
    } catch (err) {
      console.warn(`Vision model ${model} error, trying alternative:`, err.message);
      lastError = err;
    }
  }
  throw lastError || new Error('Vision-Erkennung fehlgeschlagen');
}

// Gemini Text & Multimodal Chat Endpoint (Text + optionale Bilder z. B. zur Materialerkennung)
app.post('/api/gemini/text', async (req, res) => {
  try {
    const { prompt, customApiKey, images } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    const ai = getGeminiClient(customApiKey);

    // If multimodal images are attached, process with gemini-3.8-flash
    if (images && Array.isArray(images) && images.length > 0) {
      const parts = [];
      for (const img of images) {
        let raw = '';
        if (typeof img === 'string') {
          raw = img;
        } else if (img && typeof img.data === 'string') {
          raw = img.data;
        } else if (img && typeof img.base64 === 'string') {
          raw = img.base64;
        }
        if (!raw) continue;
        const clean = raw.includes(',') ? raw.split(',')[1] : raw;
        if (!clean || clean.length < 10) continue;
        parts.push({
          inlineData: {
            mimeType: (img && img.mimeType) || 'image/jpeg',
            data: clean,
          },
        });
      }
      parts.push({ text: prompt });

      if (parts.length > 1) {
        try {
          const response = await ai.models.generateContent({
            model: 'gemini-3.8-flash',
            contents: { parts },
            config: {
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            },
          });
          if (response && response.text) {
            return res.json({ text: response.text });
          }
        } catch (err) {
          console.warn('Multimodal chat failed on gemini-3.8-flash, falling back to text prompt:', err.message);
        }
      }
      // Fallback to text prompt if multimodal failed or no valid images
      const text = await generateTextFast(ai, prompt);
      return res.json({ text: text || '' });
    }

    const text = await generateTextFast(ai, prompt);
    res.json({ text: text || '' });
  } catch (err) {
    console.error('Gemini text error:', err);
    res.status(500).json({ error: err.message || 'Auslastungsfehler bei der KI' });
  }
});

// Gemini Vision / OCR Endpoint (Blitz-Scan für 10-stellige AT-Nummer & Lieferschein-Analyse)
app.post('/api/gemini/vision', async (req, res) => {
  try {
    const { imageBase64, prompt, mimeType, customApiKey } = req.body;
    if (!imageBase64 || !prompt) {
      return res.status(400).json({ error: 'Image and prompt are required' });
    }
    const cleanBase64 = imageBase64.includes(',') ? imageBase64.split(',')[1] : imageBase64;
    const ai = getGeminiClient(customApiKey);
    const text = await generateVisionFast(ai, cleanBase64, prompt, mimeType || 'image/jpeg');
    res.json({ text: text || '' });
  } catch (err) {
    console.error('Gemini vision error:', err);
    res.status(500).json({ error: err.message || 'Vision-Fehler' });
  }
});

// Serve static assets from project root
app.use(express.static(__dirname));

// Fallback all routes to index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`Lagerratten-App running on http://${HOST}:${PORT}`);
});
