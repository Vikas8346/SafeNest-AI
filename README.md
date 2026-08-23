# SafeNest AI — Web Dashboard (Phase 1)

A browser-based version of the Phase 1 detector: a Flask backend runs the
YOLO model, and a dashboard page uses your browser's webcam (via
`getUserMedia`), sends frames to the backend for detection, and draws a
HUD-style overlay with a live risk banner, detection list, and event log.

Tested end-to-end before delivery: server boots, `/detect` returns real
detections, and once warmed up inference runs at roughly 150–200ms/frame
on CPU — fast enough for a smooth live feed.

## Files

```
webapp/
├── app.py              # Flask backend + /detect API
├── config.py           # class lists, danger weights, risk thresholds (same as Phase 1)
├── requirements.txt
├── templates/
│   └── index.html      # dashboard page
└── static/
    ├── style.css        # console/HUD styling
    └── script.js         # webcam capture, API calls, canvas overlay
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Emergency calling with Twilio

Emergency calling is configured on the backend. Set these PowerShell variables
before starting the server; never put the Twilio auth token in browser code:

```powershell
$env:TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN = "your-auth-token"
$env:TWILIO_FROM_NUMBER = "+15551234567"       # Twilio number
$env:FAMILY_PHONE_NUMBER = "+15557654321"      # family member's number
```

When the risk engine reaches `CRITICAL`, the app speaks “Papa, please help”,
saves the alert image and risk data in `saved_alerts/`, and places one Twilio
voice call per incident. The Twilio account must have voice enabled and the
destination number must be verified when using a trial account. Without these
variables, detection and local capture still work, but the call reports that
Twilio is not configured.

## Run it

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. Your browser will ask
for camera permission — allow it. First request after startup takes ~1–2
seconds (model warm-up); after that it's fast.

> Note: browsers only grant camera access on `localhost` or over HTTPS, so
> if you deploy this anywhere other than your own machine you'll need TLS.

## How it works

1. `script.js` grabs a frame from your webcam every 700ms and POSTs it as a
   JPEG data URL to `/detect`.
2. `app.py` decodes it, runs the YOLO model, and returns JSON: each
   detection's class/confidence/box (normalized 0–1 so it scales to any
   canvas size), plus a computed risk score.
3. The browser draws HUD-style corner-bracket boxes on a canvas overlaid on
   the video (blue = person, red = dangerous object, gray = neutral object),
   updates the risk banner color, and logs HIGH RISK / CRITICAL transitions
   to the event log panel.

## Tuning

- `SEND_INTERVAL_MS` in `static/script.js` controls how often frames are
  sent — lower it for a snappier feed at the cost of more server load.
- `DANGEROUS_CLASSES`, `DIST_HIGH_RISK_PX`, `DIST_WARNING_PX` in
  `config.py` control which objects count as dangerous and how proximity
  maps to risk — same file as the Phase 1 desktop version, so changes
  apply to both.

## Same Phase 1 limitations apply

No child/adult distinction, no multi-frame tracking/duration confirmation,
and the risk score is a placeholder formula rather than a trained model —
see the main Phase 1 README for the full roadmap (Phases 2–10).
