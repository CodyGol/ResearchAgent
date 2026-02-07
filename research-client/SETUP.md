# Deep Research Console Setup

## Environment Configuration

Create a `.env.local` file in the `research-client` directory with:

```env
NEXT_PUBLIC_BACKEND_URL=https://research-agent-v2-69957378560.us-central1.run.app
```

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Create `.env.local` (see above)

3. Run development server:
```bash
npm run dev
```

4. Open http://localhost:3000

## Features

- **Non-blocking UI**: Handles long-running requests (up to 10 minutes) without freezing
- **Accurate Timer**: Uses `requestAnimationFrame` to prevent browser throttling
- **Error Handling**: Robust error boundaries and timeout handling
- **Generative UI**: Smooth fade-in animations for results
- **Terminal Aesthetic**: Black background with green monospace font
