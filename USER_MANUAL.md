# AtmosCare User Manual

**Version 1.0** · Smart Air Quality Prediction

AtmosCare helps you monitor air pollution (smog), view AI-powered forecasts, explore maps, and receive alerts when air quality or weather conditions change. This guide explains every screen and feature from a user's perspective.

---

## Table of Contents

1. [What is AtmosCare?](#1-what-is-atmoscare)
2. [Getting Started](#2-getting-started)
3. [Navigation Overview](#3-navigation-overview)
4. [Dashboard](#4-dashboard)
5. [Settings](#5-settings)
6. [Profile](#6-profile)
7. [Analytics](#7-analytics)
8. [Locations & Map](#8-locations--map)
9. [Notifications & Alerts](#9-notifications--alerts)
10. [Admin & Authority Panel](#10-admin--authority-panel)
11. [Understanding Air Quality](#11-understanding-air-quality)
12. [Troubleshooting](#12-troubleshooting)
13. [Running the App (Setup)](#13-running-the-app-setup)

---

## 1. What is AtmosCare?

AtmosCare combines:

- **Live air quality data** from sensors or external APIs (PM2.5, PM10, ozone, NO₂, CO, temperature, humidity, wind)
- **AI forecasts** using a hybrid **GRU + SARIMA** machine-learning stack
- **Personal alerts** for smog, rain, and snow
- **City-wide advisories** sent by administrators
- **Maps and analytics** to explore pollution across Pakistan and worldwide

The app has three types of users:

| Role | Who | What you can do |
|------|-----|-----------------|
| **User** | Everyone after sign-up | Dashboard, Analytics, Locations, Settings, Profile |
| **Authority** | Promoted by admin | Above + send city broadcasts, view users/devices |
| **Admin** | First registered user (auto) or promoted | Full control: roles, delete users, device flags, audit log |

---

## 2. Getting Started

### 2.1 Splash Screen

When you open the app, you see the **AtmosCare** splash screen with the tagline *“Breathe smarter. Live healthier.”* The app loads for about **10 seconds**, then takes you to the **Login** screen.

### 2.2 Create an Account (Sign Up)

1. On the Login screen, tap **Don't have an account? Sign Up**
2. Fill in:
   - **Username**
   - **Email** (must be valid, e.g. `you@example.com`)
   - **Password**
3. Tap **Create Account**
4. On success, you are returned to **Login**

> **Note:** The **first person** to register on a new database automatically becomes **Admin**.

### 2.3 Sign In

1. Enter your **Email** and **Password**
2. Tap **Login** (or press Enter on the password field)
3. On success, you go to the **Dashboard**

If your location was never set, the app may try to detect it automatically from your network.

### 2.4 Sign Out

You can log out from:

- The **hamburger menu** → **Logout**
- **Profile** → **Logout**
- **Admin Panel** → logout icon (top right)

Logging out stops background alerts and returns you to the Login screen.

---

## 3. Navigation Overview

### Bottom navigation bar

Available on main screens (Dashboard, Settings, Profile, etc.):

| Tab | Goes to |
|-----|---------|
| **Dashboard** | Home screen with live AQI and forecasts |
| **Analytics** | Trends, pollution sources, health advice |
| **Locations** | Map + Pakistan city smog table |
| **Settings** | Name, location, alert preferences |
| **Admin** | Admin/Authority panel *(only if you have that role)* |

### Hamburger menu (Dashboard)

Open with the **☰ menu** icon on the Dashboard top bar:

- Dashboard
- Analytics
- Location
- Settings
- Admin Panel / Authority Panel *(role-dependent)*
- Logout

### Dashboard top bar icons

| Icon | Action |
|------|--------|
| **☰ Menu** | Open side drawer |
| **↻ Refresh** | Reload dashboard data |
| **👤 Account** | Open Profile |
| **🔔 Bell** | Open notification history |

---

## 4. Dashboard

The Dashboard is your home screen. It shows your location, current air quality, model predictions, and health advice.

### 4.1 Location

- Your city appears at the top of the hero card (e.g. `Lahore, Pakistan`)
- Tap the **pencil icon** next to the location to open **Settings** and change it
- **Last updated** time shows when data was last refreshed

### 4.2 Data source badge

A colored badge shows where data comes from:

| Badge | Meaning |
|-------|---------|
| **Live IoT Sensor** (green) | Reading from a connected air-quality sensor |
| **External API Data** (amber) | Live data from an external air-quality API |
| **IoT + API Hybrid** (green) | Sensor data supplemented by API |

Data refreshes automatically about every **15 seconds** while the Dashboard is open. Use the **refresh** icon for an immediate update.

### 4.3 Hero card — Predicted smog (PM2.5)

The large gauge shows your **predicted PM2.5** level (fine particulate matter — the main smog indicator):

- The gauge color matches the air quality category (Good → Hazardous)
- A **segmented scale** below shows: Good, Mod, USG, Unh, Very, Hzd
- **Confidence** may be shown when the ML model provides it

### 4.4 Realtime pollutants

The **Realtime Pollutants** card lists current readings:

| Pollutant | Description |
|-----------|-------------|
| **PM2.5** | Fine particles (main smog measure) |
| **PM10** | Coarse particles |
| **O3** | Ozone |
| **NO2** | Nitrogen dioxide |
| **CO** | Carbon monoxide |
| **Smog Index** | Combined pollution score |
| **Temp / Humidity / Wind** | Weather conditions |

Tags like `(sensor)`, `(API)`, or `(est.)` show the data source.

### 4.5 Forecast risk

The **Forecast Risk** card shows ML model predictions:

- **Tomorrow** — next-day PM2.5 estimate
- **7-day avg** — average over the next week
- **Day 7 est.** — estimate for day 7

Each value includes a category badge (Good, Moderate, Unhealthy, etc.) and a risk bar.

### 4.6 Trend snapshot

Short summaries such as *“7 days: …”* and *“30 days: …”* give a quick view of expected trends.

### 4.7 Health advisory

A card shows plain-language advice based on current conditions, for example:

- *“High pollution levels detected. Limit outdoor activity.”* (AQI > 150)
- *“Moderate pollution. Sensitive groups should take caution.”* (AQI > 100)
- *“Air quality is acceptable for most people.”* (otherwise)

This is informational text on the Dashboard; **push-style alerts** are separate (see [Section 9](#9-notifications--alerts)).

---

## 5. Settings

Open **Settings** from the bottom bar, hamburger menu, or the location pencil on the Dashboard.

### 5.1 Personal info

| Field | Purpose |
|-------|---------|
| **Name** | Display name on your Profile |
| **Location** | City used for forecasts and weather alerts (e.g. `Lahore, Pakistan`) |
| **Detect** | Auto-detect location via GPS (mobile) or IP/network |

**Tip:** Use a real city name. Alerts and predictions use the first part of the location (before the comma).

### 5.2 Alert preferences

Turn notifications on or off:

| Alert | Default | What it does |
|-------|---------|--------------|
| **Smog Level Alerts** | On | Warns when PM2.5 or AQI crosses unhealthy thresholds |
| **Rain Alerts** | Off | Warns when rain is forecast (Open-Meteo, next 24 hours) |
| **Snow Alerts** | Off | Warns when snow is forecast (Open-Meteo, next 24 hours) |

### 5.3 Save changes

Tap **Save Changes** to store your settings. They are saved to your account and used immediately for alerts and predictions.

### 5.4 Location permission (Android)

On first launch, Android may ask **Allow location access**. Choosing **Allow** improves auto-detect; **Deny** still lets you type a city manually.

---

## 6. Profile

Open Profile from the **account icon** on the Dashboard or via navigation.

### What you see

- **Name** and **email**
- **Role badge** — User, Authority, or Admin
- **Location** — your saved city
- **Rain / Snow** — whether those weather alerts are enabled (Yes/No)

### Actions

| Button | Action |
|--------|--------|
| **Edit Profile** | Opens Settings |
| **Logout** | Signs you out |
| **← Back** | Returns to Dashboard |

---

## 7. Analytics

Open **Analytics** from the bottom bar or menu.

### 7.1 Toolbar

| Button | Action |
|--------|--------|
| **This Week** | Range label (display) |
| **Export** | Saves data to `analytics_export.csv` in the app folder |
| **Share** | Copies a text summary to the clipboard |
| **↻ Refresh** | Reloads analytics |

### 7.2 Tabs

#### Trends (default)

- **Current air quality** — today’s gauge, PM2.5 or AQI, status, and health text
- **Daily forecast** — up to 7 day cards with PM2.5, status, and *“GRU+SARIMA stacked”* label

#### Sources

**SHAP Source Attribution** — shows what drives the model’s PM2.5 forecast:

- Traffic emissions
- Industrial activity
- Weather / other
- Crop burning

Bars and an insight line explain the top contributing factors.

#### Health

Recommendations based on predicted smog:

- Sensitive groups (children, elderly, asthma)
- General public
- Exercise guidance

---

## 8. Locations & Map

### 8.1 Global map

- **Pan and zoom** the map (drag, pinch, or use **−** / **+** buttons)
- **Colored dots** = air quality monitoring stations (WAQI data)
- **Legend:** Good · Mod · USG · Unh · V.Unh · Hzd
- Tap **Refresh** to reload stations
- Map centers on your GPS, IP location, or Settings city

### 8.2 Pakistan smog levels

A table of **8 major cities**:

Lahore, Karachi, Islamabad, Peshawar, Quetta, Multan, Faisalabad, Rawalpindi

Each row shows predicted PM2.5 (or smog index), status, and color. Footer shows **Last updated** time. Tap **Refresh** to update.

---

## 9. Notifications & Alerts

### 9.1 Viewing notifications

Tap the **bell icon** on the Dashboard.

- Lists up to **20 recent alerts** (time, title, message)
- Opening the list **marks all as read** (bell icon changes from ringing to normal)
- If no alerts yet, you may see the current health advisory and a tip to enable alerts in Settings

### 9.2 Smog alerts

Requires **Smog Level Alerts** enabled in Settings. Checked every **60 seconds** and on live dashboard updates.

**PM2.5 thresholds (µg/m³):**

| PM2.5 | Alert type | How shown |
|-------|------------|-----------|
| > 150 | Severe | Pop-up dialog — *“Stay indoors”* |
| > 55 | Unhealthy | Pop-up dialog — *“Limit outdoor activity”* |
| > 35 | Moderate | Bottom snackbar notice |
| 15% rise above previous (if > 35) | Increase | Dialog — *“Smog Level Increased”* |

**AQI fallback** (when PM2.5 is unavailable):

| AQI | Alert |
|-----|-------|
| > 200 | Severe dialog |
| > 150 | Unhealthy dialog |
| > 100 | Moderate snackbar |

Repeat alerts for the same level are limited to once every **30 minutes**.

### 9.3 Rain and snow alerts

Requires **Rain Alerts** or **Snow Alerts** in Settings.

- Uses **Open-Meteo** forecast for your Settings city (next 24 hours)
- First detection → pop-up **Rain Alert** or **Snow Alert**
- Alert clears when the condition is no longer forecast

### 9.4 City advisories (broadcasts)

Administrators and authorities can send **city advisories**. These appear as pop-up dialogs with a custom title and message. Targeted broadcasts show the city name, e.g. `[Lahore] Stay indoors today.` Advisories stay active for **24 hours**.

---

## 10. Admin & Authority Panel

**Access:** Only **Admin** and **Authority** roles. Others see *“Access denied”* and return to the Dashboard.

### 10.1 Header

- Title: **Admin Panel** or **Authority Panel**
- Stats: **Users**, **Devices**, **Cities**
- **Refresh** and **Logout** icons

### 10.2 Tabs

| Tab | Admin | Authority |
|-----|-------|-----------|
| **Overview** | System stats, top cities | Same |
| **Users** | Search, change role, delete | View only |
| **Devices** | Enable/disable, force API, mark test | View only |
| **Broadcast** | Send + view recent advisories | Send + view recent |
| **Audit** | Full action log | Hidden (admins only) |

### 10.3 Overview

- User counts by role
- IoT device totals
- ML engine status (GRU, SARIMA, stacking loaded or not)
- Database health
- Top cities by number of users

### 10.4 Users (admin only for changes)

- Search by email, name, or city
- **Change Role** — set User, Authority, or Admin
- **Delete** — requires your admin password; cannot delete yourself or the last admin

### 10.5 Devices (admin only for toggles)

Per IoT device you can see ID, last seen, PM readings, location, and buffer size.

| Action | Purpose |
|--------|---------|
| **Disable / Enable** | Turn device off/on |
| **Force API / Use Sensor** | Override data source |
| **Mark Test / Unmark Test** | Flag as test device |

### 10.6 Broadcast

**Send City Advisory** form:

| Field | Description |
|-------|-------------|
| **City** | Target city, or `*` for all cities |
| **Alert title** | Short headline |
| **Advisory message** | Full message text |
| **Send Broadcast** | Publishes for 24 hours |

**Recent Broadcasts** lists active and expired advisories.

### 10.7 Audit (admin only)

Chronological log of admin actions: timestamp, action, actor, target, details.

---

## 11. Understanding Air Quality

### PM2.5 (primary smog measure)

Fine particles smaller than 2.5 micrometers. Lower is better.

| PM2.5 (µg/m³) | Category | General advice |
|---------------|----------|----------------|
| 0–12 | Good | Safe for most outdoor activity |
| 12–35 | Moderate | Acceptable; sensitive people may notice |
| 35–55 | Unhealthy for sensitive groups | Limit long outdoor exertion if sensitive |
| 55–150 | Unhealthy | Everyone should reduce outdoor activity |
| 150+ | Very unhealthy / Hazardous | Stay indoors; use masks if you must go out |

### AQI scale (dashboard segments)

| Label | Meaning |
|-------|---------|
| **Good** | 0–50 |
| **Mod** (Moderate) | 51–100 |
| **USG** (Unhealthy for Sensitive Groups) | 101–150 |
| **Unh** (Unhealthy) | 151–200 |
| **Very** (Very Unhealthy) | 201–300 |
| **Hzd** (Hazardous) | 301+ |

### How predictions work

AtmosCare uses a **hybrid AI model**:

1. **GRU** neural network — learns patterns from recent pollution and weather
2. **SARIMA** — captures seasonal trends
3. **Stacking (XGBoost)** — combines both into a final PM2.5 forecast

Forecasts appear on the Dashboard and in Analytics.

---

## 12. Troubleshooting

### “Database connection failed” on sign-up or login

- MongoDB must be running and reachable
- For local use: start MongoDB on `mongodb://localhost:27017`
- For cloud: set `DATABASE_URI` to your MongoDB Atlas connection string

### Dashboard shows “Unable to fetch AQI data”

- Check that the **backend server** is running (default `http://127.0.0.1:8000`)
- Confirm your **location** is set in Settings (not `Unknown Location`)
- Tap **Refresh** on the Dashboard

### No smog / rain / snow alerts

1. Open **Settings** → enable the relevant alert checkboxes → **Save Changes**
2. Set a valid **location** (city name)
3. Wait up to **60 seconds** for the alert poll, or refresh the Dashboard
4. Smog alerts need PM2.5 or AQI data above thresholds

### Map shows no stations

- Check internet connection
- Tap **Refresh** on the Locations screen
- Some regions have fewer public monitoring stations

### Location not detected

- On Android, allow location permission when prompted
- Or type your city manually in Settings (e.g. `Karachi, Pakistan`)

### Admin panel says “Access denied”

- Your account role is **User**. Ask an admin to promote you to **Authority** or **Admin**.

### Backend health check

If you host the API, open:

```
https://YOUR-BACKEND-URL/health
```

A healthy response includes `"database": "ok"` and model flags (`gru_loaded`, `sarima_loaded`, `stacking_loaded`).

---

## 13. Running the App (Setup)

*For installers and developers — end users typically receive a pre-built app or installer.*

### Requirements

- Python 3.11+
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/atlas))
- Internet for air-quality APIs and weather forecasts

### Quick start (local)

**1. Start MongoDB** (or use Atlas URI in environment)

**2. Start the backend**

```bash
cd AtmosCare-main
pip install -r requirements-backend.txt
uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

**3. Start the frontend**

```bash
cd Frontend
pip install -r ../requirements.txt
python main.py
```

**4. First use**

- Sign up → first user becomes **Admin**
- Log in → set location and alerts in **Settings**

### Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `DATABASE_URI` | MongoDB connection string |
| `DATABASE_NAME` | Database name (default `AtmosCareDB`) |
| `BACKEND_URL` | API URL for the app (default `http://127.0.0.1:8000`) |
| `WAQI_API_KEY` | World Air Quality Index API key |
| `GOOGLE_API_KEY` | Google Maps (optional, for map tiles) |

See `Backend/example.env` for a template.

### Cloud deployment

The backend can be deployed with Docker (`Dockerfile`), Render, Fly.io, or similar. Point the app’s `BACKEND_URL` at your deployed API. MongoDB Atlas is recommended for the database.

---

## Quick reference card

| I want to… | Go to… |
|------------|--------|
| See current air quality | **Dashboard** |
| Change my city | **Settings** → Location → **Save Changes** |
| Turn on rain/snow alerts | **Settings** → checkboxes → **Save Changes** |
| View alert history | Dashboard **bell icon** |
| See 7-day forecast | **Analytics** → **Trends** |
| See pollution sources | **Analytics** → **Sources** |
| Explore a map | **Locations** |
| Send a city warning | **Admin Panel** → **Broadcast** *(admin/authority)* |
| Manage users | **Admin Panel** → **Users** *(admin)* |
| Log out | **Profile** or menu → **Logout** |

---

*AtmosCare v1.0 — Breathe smarter. Live healthier.*
