# AtmosCare — Portable Setup (Any PC + Android APK)

## Quick start on any Windows PC

1. Install **Python 3.11+** and ensure it is on PATH.
2. Open PowerShell in this folder:

```powershell
.\setup_portable.ps1
.\run_atmoscare.ps1
```

That will:

- Create `.env` with **MongoDB Atlas** (no local MongoDB install)
- Install dependencies into `.venv`
- Start the FastAPI backend on `http://127.0.0.1:8000`
- Launch the Kivy desktop app

### Database

Default Atlas URI is already in `portable.env` / `Backend/config.py`.

```
DATABASE_URI=mongodb+srv://...@cluster0.4ruvvwt.mongodb.net/...
DATABASE_NAME=AtmosCareDB
```

Atlas Network Access must allow your IP (or `0.0.0.0/0`).

---

## Android APK

### Build (GitHub Actions — recommended)

1. Push to `main` (workflow: `.github/workflows/build-apk.yml`).
2. Download artifact **atmoscare-debug-apk** from the Actions run.
3. Install the APK on your phone.

App icon and launch splash use `Frontend/assets/logo.png` → `icon.png` / `presplash.png`.

### Important: phone must reach the backend

The APK does **not** bundle MongoDB or TensorFlow. It talks to the backend over HTTP.

Before building (or after install by editing/rebuilding), set your public or LAN backend URL in:

`Frontend/config.json`

```json
{
  "BACKEND_URL": "http://YOUR_PC_LAN_IP:8000"
}
```

Examples:

- Same Wi‑Fi as PC: `http://192.168.1.25:8000`
- Cloud host (Render / Railway / VPS): `https://your-api.example.com`

Then rebuild the APK so the URL is baked in.

Windows firewall: allow inbound **TCP 8000** when using LAN IP.

---

## Manual run (without scripts)

```powershell
copy portable.env .env
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements-backend.txt
pip install -r requirements.txt
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
# new terminal
cd Frontend
python main.py
```
