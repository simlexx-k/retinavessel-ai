# RetinaVessel frontend

Next.js App Router and TypeScript interface for the separate FastAPI inference
service. Browser requests use same-origin route handlers; the backend URL and
optional API key remain server-only.

The interface is organized as an imaging workstation: validated drag-and-drop
upload, registered-model provenance, threshold and overlay controls, an
original/result comparison viewport, zoom controls, inference telemetry and
PNG export. Its UI stack uses Radix UI primitives, TanStack Query,
react-dropzone, React Compare Slider, Motion, Lucide and Sonner. These packages
are pinned in `package.json` and `package-lock.json`.

## Local development

Start the FastAPI backend first, then:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

Open <http://127.0.0.1:3000>. The default server-side backend URL is
`http://127.0.0.1:8000`.

## Verification

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## Full Docker stack

From the repository root, after creating `backend/model_registry.json` and
placing checkpoints under `models/`:

```bash
docker compose up -d --build
```

The web interface is bound to `127.0.0.1:3000`; FastAPI remains available at
`127.0.0.1:8000`. Set the same `RETINAVESSEL_API_KEY` for both services when
API-key protection is enabled.

## Boundary

This interface displays research segmentation output. It is not intended for
diagnosis, treatment, triage, or clinical decision-making.
