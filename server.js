import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import Stripe from 'stripe';
import nodemailer from 'nodemailer';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createHmac, createHash, randomBytes } from 'crypto';
import cookieParser from 'cookie-parser';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';

dotenv.config();

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const stripe  = new Stripe(process.env.STRIPE_SECRET_KEY);

const COOKIE_SECRET = process.env.COOKIE_SECRET || (() => {
  const s = randomBytes(32).toString('hex');
  console.warn('\n  ⚠  COOKIE_SECRET not set — sessions will not survive restarts.\n');
  return s;
})();

app.use(express.json());
app.use(cookieParser());

// ── Data persistence ────────────────────────────────────────────────────────
const DATA_DIR = join(__dirname, 'data');
if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
const AGENTS_FILE = join(DATA_DIR, 'agents.json');

function loadAgents() {
  if (!existsSync(AGENTS_FILE)) return {};
  try { return JSON.parse(readFileSync(AGENTS_FILE, 'utf8')); } catch { return {}; }
}

function saveAgents(data) {
  writeFileSync(AGENTS_FILE, JSON.stringify(data, null, 2));
}

function getConfig(sessionKey) {
  const agents = loadAgents();
  if (agents[sessionKey]) return agents[sessionKey];
  // First time — create default config with stable agentId
  const config = {
    agentId:      randomBytes(8).toString('hex'),
    businessName: '',
    tagline:      '',
    description:  '',
    specialties:  '',
    alertEmail:   '',
    trips:        [],
    faqs:         [],
  };
  agents[sessionKey] = config;
  saveAgents(agents);
  return config;
}

function saveConfig(sessionKey, updates) {
  const agents = loadAgents();
  const existing = agents[sessionKey] || getConfig(sessionKey);
  agents[sessionKey] = { ...existing, ...updates, agentId: existing.agentId };
  saveAgents(agents);
  return agents[sessionKey];
}

function getConfigByAgentId(agentId) {
  const agents = loadAgents();
  return Object.values(agents).find(a => a.agentId === agentId) || null;
}

// ── Email alerts ────────────────────────────────────────────────────────────
const mailer = nodemailer.createTransport({
  host:   process.env.SMTP_HOST || 'smtp.gmail.com',
  port:   parseInt(process.env.SMTP_PORT || '587'),
  secure: false,
  auth:   { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
});

async function sendAlert(alertEmail, clientMessage, alertReason, businessName) {
  if (!process.env.SMTP_USER || !alertEmail) return;
  try {
    await mailer.sendMail({
      from:    `TravelDesk <${process.env.SMTP_USER}>`,
      to:      alertEmail,
      subject: `TravelDesk: A client needs your attention`,
      html: `
        <div style="font-family:system-ui,sans-serif;max-width:580px;margin:0 auto;color:#111827">
          <div style="background:#2563eb;padding:20px 24px;border-radius:10px 10px 0 0">
            <h2 style="color:#fff;margin:0;font-size:18px">✈️ TravelDesk Alert</h2>
          </div>
          <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;padding:24px;border-radius:0 0 10px 10px">
            <p style="margin:0 0 16px">A client is asking something that needs your personal attention at <strong>${businessName || 'your agency'}</strong>.</p>
            <div style="background:#f8fafc;border-left:3px solid #2563eb;padding:14px 16px;border-radius:4px;margin-bottom:16px">
              <p style="margin:0;font-size:14px;color:#374151"><strong>Client's message:</strong></p>
              <p style="margin:8px 0 0;font-size:15px">${clientMessage}</p>
            </div>
            <div style="background:#fef3c7;border:1px solid #fcd34d;padding:12px 16px;border-radius:6px">
              <p style="margin:0;font-size:14px"><strong>Flagged because:</strong> ${alertReason}</p>
            </div>
            <p style="margin:20px 0 0;font-size:13px;color:#6b7280">Log in to your TravelDesk dashboard to follow up with this client.</p>
          </div>
        </div>
      `,
    });
  } catch (err) {
    console.error('Email alert failed:', err.message);
  }
}

// ── System prompt builder ───────────────────────────────────────────────────
function buildSystemPrompt(config) {
  const biz = config.businessName || 'this travel agency';

  let p = `You are a friendly, knowledgeable AI travel assistant for ${biz}. You help clients with travel questions, provide information about their trips, and assist with general travel enquiries.`;

  if (config.tagline) p += ` ${config.tagline}.`;

  if (config.description) p += `\n\nAbout ${biz}:\n${config.description}`;

  if (config.specialties) p += `\n\nSpecialities and expertise:\n${config.specialties}`;

  if (config.trips?.length > 0) {
    p += '\n\nClient trips on file:\n';
    config.trips.forEach((t, i) => {
      let line = `${i + 1}. `;
      if (t.clientName)  line += `Client: ${t.clientName} — `;
      if (t.destination) line += `${t.destination}`;
      if (t.dates)       line += `, ${t.dates}`;
      if (t.bookingRef)  line += ` (Ref: ${t.bookingRef})`;
      if (t.notes)       line += `. Notes: ${t.notes}`;
      p += line + '\n';
    });
  }

  if (config.faqs?.length > 0) {
    p += '\n\nFrequently asked questions for this agency:\n';
    config.faqs.forEach(f => {
      if (f.question && f.answer) p += `Q: ${f.question}\nA: ${f.answer}\n\n`;
    });
  }

  p += `\n\nBehaviour rules:
- Be warm, helpful, and concise
- Use the trip and FAQ data above when relevant
- For general travel knowledge (visas, packing, destinations, weather) use your expertise
- If a question requires the agent's direct involvement — specific pricing, booking changes, complaints, or anything you cannot confidently answer — end your response with exactly this on its own line: <<<ALERT:one-sentence reason>>>
- Never fabricate booking details, prices, or specific dates you don't have`;

  return p;
}

// ── Auth middleware ─────────────────────────────────────────────────────────
function isLocalhost(req) {
  const ip = req.ip || req.connection?.remoteAddress || '';
  return ip === '127.0.0.1' || ip === '::1' || ip === '::ffff:127.0.0.1';
}

function requireAuth(req, res, next) {
  if (isLocalhost(req)) {
    req.isLocalhost = true;
    req.sessionKey  = 'localhost-dev';
    return next();
  }
  const token = req.cookies?.td_session;
  if (!token) return res.redirect('/');
  try {
    const dot = token.lastIndexOf('.');
    if (dot === -1) throw new Error('malformed');
    const data = token.slice(0, dot);
    const sig  = token.slice(dot + 1);
    const expected = createHmac('sha256', COOKIE_SECRET).update(data).digest('base64url');
    if (sig !== expected) throw new Error('bad sig');
    const { exp } = JSON.parse(Buffer.from(data, 'base64url').toString());
    if (Date.now() > exp) throw new Error('expired');
    req.sessionKey  = createHash('sha256').update(data).digest('hex').slice(0, 16);
    req.isLocalhost = false;
    next();
  } catch {
    res.clearCookie('td_session');
    res.redirect('/');
  }
}

// ── Pages ───────────────────────────────────────────────────────────────────
app.get('/', (req, res) =>
  res.sendFile(join(__dirname, 'public', 'landing.html'))
);
app.get('/dashboard', requireAuth, (req, res) =>
  res.sendFile(join(__dirname, 'public', 'dashboard.html'))
);
app.get('/chat', (req, res) =>
  res.sendFile(join(__dirname, 'public', 'chat.html'))
);

// ── Stripe ──────────────────────────────────────────────────────────────────
app.post('/api/create-checkout-session', async (req, res) => {
  try {
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      line_items: [{ price: process.env.STRIPE_PRICE_ID, quantity: 1 }],
      subscription_data: { trial_period_days: 7 },
      success_url: `${req.protocol}://${req.get('host')}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url:  `${req.protocol}://${req.get('host')}/`,
    });
    res.json({ url: session.url });
  } catch (err) {
    console.error('Stripe error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/success', async (req, res) => {
  const { session_id } = req.query;
  if (!session_id) return res.redirect('/');
  try {
    const session = await stripe.checkout.sessions.retrieve(session_id);
    if (session.status !== 'complete') return res.redirect('/');
    const email = session.customer_details?.email || '';
    const exp   = Date.now() + 30 * 24 * 60 * 60 * 1000;
    const data  = Buffer.from(JSON.stringify({ email, exp })).toString('base64url');
    const sig   = createHmac('sha256', COOKIE_SECRET).update(data).digest('base64url');
    res.cookie('td_session', `${data}.${sig}`, {
      httpOnly: true,
      secure:   process.env.NODE_ENV === 'production',
      maxAge:   30 * 24 * 60 * 60 * 1000,
      sameSite: 'lax',
    });
    res.redirect('/dashboard');
  } catch (err) {
    console.error('Success handler error:', err);
    res.redirect('/');
  }
});

// ── Admin bypass ────────────────────────────────────────────────────────────
app.get('/admin', (req, res) => {
  const key = process.env.ADMIN_SECRET_KEY;
  if (!key || req.query.key !== key) return res.status(403).send('Forbidden');
  const exp  = Date.now() + 365 * 24 * 60 * 60 * 1000;
  const data = Buffer.from(JSON.stringify({ email: 'admin', exp })).toString('base64url');
  const sig  = createHmac('sha256', COOKIE_SECRET).update(data).digest('base64url');
  res.cookie('td_session', `${data}.${sig}`, {
    httpOnly: true,
    secure:   process.env.NODE_ENV === 'production',
    maxAge:   365 * 24 * 60 * 60 * 1000,
    sameSite: 'lax',
  });
  res.redirect('/dashboard');
});

// ── Agent config API ─────────────────────────────────────────────────────────
app.get('/api/config', requireAuth, (req, res) => {
  res.json(getConfig(req.sessionKey));
});

app.put('/api/config', requireAuth, (req, res) => {
  const allowed = ['businessName','tagline','description','specialties','alertEmail','trips','faqs'];
  const updates = {};
  allowed.forEach(k => { if (req.body[k] !== undefined) updates[k] = req.body[k]; });
  res.json(saveConfig(req.sessionKey, updates));
});

// ── Public agent info (for chat widget) ─────────────────────────────────────
app.get('/api/agent/:agentId', (req, res) => {
  const config = getConfigByAgentId(req.params.agentId);
  if (!config) return res.status(404).json({ error: 'Agent not found' });
  res.json({ businessName: config.businessName, tagline: config.tagline });
});

// ── Chat (public, SSE streaming) ─────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  const { agentId, message, history = [] } = req.body;

  if (!agentId || !message?.trim()) {
    return res.status(400).json({ error: 'agentId and message are required.' });
  }

  // localhost dev: use a generic config
  let config;
  if (agentId === 'dev') {
    config = { businessName: 'Dev Agency', trips: [], faqs: [], alertEmail: '' };
  } else {
    config = getConfigByAgentId(agentId);
    if (!config) return res.status(404).json({ error: 'Agent not found.' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const systemPrompt = buildSystemPrompt(config);

  // Build messages array from history + new message
  const messages = [
    ...history.slice(-10).map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message.trim() },
  ];

  try {
    const stream = client.messages.stream({
      model: 'claude-opus-4-7',
      max_tokens: 1024,
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      messages,
    });

    let fullText = '';
    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
        fullText += event.delta.text;
        res.write(`data: ${JSON.stringify({ text: event.delta.text })}\n\n`);
      }
    }

    // Detect and handle alert
    const alertMatch = fullText.match(/<<<ALERT:([^>]*)>>>/);
    const alerted = !!alertMatch;
    if (alerted) {
      const reason = alertMatch[1].trim();
      await sendAlert(config.alertEmail, message.trim(), reason, config.businessName);
    }

    res.write(`data: ${JSON.stringify({ done: true, alerted })}\n\n`);
    res.end();
  } catch (err) {
    console.error('Anthropic error:', err);
    const msg = err?.status === 401
      ? 'Invalid API key.'
      : err?.message || 'Something went wrong.';
    res.write(`data: ${JSON.stringify({ error: msg })}\n\n`);
    res.end();
  }
});

const PORT = process.env.PORT || 3002;
app.listen(PORT, () => {
  console.log(`\n  TravelDesk → http://localhost:${PORT}\n`);
});
