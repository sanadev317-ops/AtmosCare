[app]

# (str) Title of your application
title = AtmosCare

# (str) Package name
package.name = atmoscare

# (str) Package domain (needed for android/ios packaging)
package.domain = org.atmoscare

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,jpeg,kv,atlas,ttf,otf,json

# (list) List of inclusions using pattern matching
source.include_patterns = assets/*, assets/*/*

# (list) Source files to exclude (let empty to not exclude anything)
source.exclude_exts = spec

# (list) List of directory to exclude (let empty to not exclude anything)
source.exclude_dirs = tests, bin, venv, __pycache__, .buildozer, .history

# (str) Application versioning (method 1)
version = 1.0

# (list) Application requirements
# MODIFIED: Removed version pins from kivy and kivymd. Buildozer needs to fetch matching recipes natively.
requirements = python3,kivy,kivymd,requests,plyer,urllib3,certifi,charset-normalizer,idna,pillow

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/assets/logo.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/assets/logo.png

# (list) Supported orientations
# Valid options are: landscape, portrait, portrait-reverse, landscape-reverse
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Permissions
android.permissions = INTERNET, ACCESS_NETWORK_STATE, ACCESS_FINE_LOCATION, ACCESS_COARSE_LOCATION

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK / AAB will support.
android.minapi = 21

# (int) Android SDK version to use
android.sdk = 33

# (str) Android NDK version to use
# MODIFIED: Changed 25b to 25c. Kivy 2.3+ compilation fails on NDK 25b due to a known toolchain bug.
android.ndk = 25c

# ADDED: Forces Buildozer to download missing components like AIDL dynamically on the GitHub runner.
android.skip_update = False

# ADDED: Forces the Android SDK manager to accept licenses automatically, resolving your previous crash.
android.accept_sdk_license = True

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (list) The Android archs to build for
android.archs = arm64-v8a, armeabi-v7a

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (str) The format used to package the app for release mode (aab or apk or aar).
android.release_artifact = apk

# (str) The format used to package the app for debug mode (apk or aar).
android.debug_artifact = apk


[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
