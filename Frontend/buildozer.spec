[app]

title = AtmosCare
package.name = atmoscare
package.domain = org.atmoscare

# Package the Frontend folder (portable APK — talks to backend over HTTP)
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,ttf,otf,json
source.include_patterns = assets/*, assets/*/*
source.exclude_exts = spec
source.exclude_dirs = tests, bin, venv, __pycache__, .buildozer, .history, cache

version = 1.0

# No TensorFlow / pymongo on device — auth + ML go through BACKEND_URL
# Pin kivy 2.3.1 for Cython 3.x compatibility on GitHub Actions
requirements = python3,kivy==2.3.1,kivymd==1.2.0,requests,plyer,urllib3,certifi,charset-normalizer,idna,pillow

presplash.filename = %(source.dir)s/assets/presplash.png
icon.filename = %(source.dir)s/assets/icon.png

orientation = portrait
fullscreen = 0

android.permissions = INTERNET, ACCESS_NETWORK_STATE, ACCESS_FINE_LOCATION, ACCESS_COARSE_LOCATION
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25c
android.skip_update = False
android.accept_sdk_license = True
android.private_storage = True
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True
android.release_artifact = apk
android.debug_artifact = apk

# Allow HTTP to LAN backend during testing
android.manifest.application_arguments = android:usesCleartextTraffic="true"

[buildozer]
log_level = 2
warn_on_root = 1
