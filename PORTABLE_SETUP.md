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

### Phone / APK — automatic IP detection

`BACKEND_URL` is set to **`auto`** by default. The app:

1. Broadcasts a LAN discovery probe (UDP **3847**)
2. Tries localhost / Android emulator host (`10.0.2.2`)
3. Tries common gateway IPs on your Wi‑Fi subnet
4. Uses the first URL that answers `/health`

Backend binds **`0.0.0.0:8000`** (all interfaces) and answers discovery probes.

**Same Wi‑Fi:** start `.\run_atmoscare.ps1` on the PC, install the APK — no manual IP needed.

Windows firewall: allow inbound **TCP 8000** and **UDP 3847**.

Optional override in `Frontend/config.json`:

```json
{ "BACKEND_URL": "auto" }
```

or a fixed cloud URL:

```json
{ "BACKEND_URL": "https://your-api.example.com" }
```

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
