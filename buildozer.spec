[app]

# (str) Title of your application
title = Chal Chitra

# (str) Package name
package.name = chalchitra

# (str) Package domain (needed for android/ios packaging)
package.domain = org.example

# (str) Source code where the main.py live
source.dir = .

# (str) Filename of the main python file
source.main_py = android_main.py

# (list) List of assets to be packaged (e.g. kv, png, jpg, etc.)
source.include_exts = py,png,jpg,kv,atlas,pt

# (list) List of exclusions (e.g. *.pyc)
source.exclude_exts = spec

# (str) Application versioning
version = 0.1

# (list) Kivy requirements
# Comma-separated list of requirements
# Requirements can be kivy, python3, sdl2, etc.
requirements = python3,kivy,plyer,opencv,numpy,mediapipe,pillow

# (str) Custom source folders for requirements
# (since you have a local folder for requirements)
# requirements.source.kivy = ../../kivy

# (str) Presplash background color (for new kivy-splash screen)
# presplash.color = #000000

# (str) Presplash animation (for new kivy-splash screen)
# presplash.filename = %(source.dir)s/data/presplash_animation.zip

# (str) Orientation (all | portrait | landscape)
orientation = landscape

# (bool) Indicate if the application should be fullscreen
fullscreen = 0


[buildozer]

# (int) Log level (0 = error, 1 = info, 2 = debug (very verbose))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1


#---------------------------------------------------------------------------
# Android specific
#---------------------------------------------------------------------------

# (list) Permissions
android.permissions = CAMERA, INTERNET

# (list) features used by the application.
# android.features = android.hardware.camera, android.hardware.camera.autofocus

# (str) Android NDK version to use
# android.ndk = 23b

# (bool) Use --private data storage (True) or --dir public storage (False)
# android.private_storage = True

# (str) Android entry point, default is org.kivy.android.PythonActivity
# android.entrypoint = org.kivy.android.PythonActivity

# (str) Android app theme, default is @android:style/Theme.NoTitleBar
# android.theme = @android:style/Theme.NoTitleBar

# (list) Pattern to whitelist for the whole project
# android.whitelist = 

# (str) Path to a custom whitelist file
# android.whitelist_src = 

# (str) Path to a custom blacklist file
# android.blacklist_src = 

# (list) List of Java files to add to the android project
# android.add_src = 

# (list) List of Java files to add to the android project (for dependencies)
# android.add_dependency_src = 

# (list) List of jars to add to the android project's libs directory
# android.add_jars = 

# (list) List of armv7-a libs to add to the android project's libs directory
# android.add_libs_armv7a = 

# (list) List of arm64-v8a libs to add to the android project's libs directory
# android.add_libs_arm64_v8a = 

# (list) List of x86 libs to add to the android project's libs directory
# android.add_libs_x86 = 

# (list) List of x86_64 libs to add to the android project's libs directory
# android.add_libs_x86_64 = 

# (bool) Copy library even if blacklisted
# android.copy_libs = True

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (int) Android API to use
android.api = 27

# (int) Minimum API required
android.minapi = 21

# (int) Target API required
android.sdk = 29

# (str) Android NDK to use for your app
# android.ndk = 19b

# (str) Android NDK path. You can specify it here instead of exporting it in your shell.
# android.ndk_path = 

# (str) Android SDK path. You can specify it here instead of exporting it in your shell.
# android.sdk_path = 

# (str) Android ANT path. You can specify it here instead of exporting it in your shell.
# android.ant_path = 

# (bool) Create a debug build of your app
# android.debug = False

# (str) Keystore path
# android.keystore.path = 

# (str) Keystore alias
# android.keystore.alias = 

# (str) Keystore password
# android.keystore.password = 

# (str) Key password
# android.keystore.key.password = 

# (bool) If true, the app will not be signed.
# android.skip_signature = False

# (str) p4a branch to use, defaults to 'master'
# p4a.branch = master

# (str) p4a directory, defaults to '~/.buildozer/android/p4a'
# p4a.dir = 

# (str) The directory in which python-for-android will be cloned
# p4a.source_dir = 

# (str) The python-for-android fork to use
# p4a.fork = kivy

# (bool) If True, p4a will not be updated
# p4a.local_recipes = 

# (str) The p4a.local_recipes path
# p4a.local_recipes_path = 

# (bool) If True, p4a will not be updated
# p4a.no_update = False

# (bool) If True, p4a will be reset before build
# p4a.reset_build = False
