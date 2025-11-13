# 🏗️ Build APK for S25 Ultra

Everything is ready. Just run these commands on your laptop:

## Step 1: Login to Expo (one-time)
```bash
cd quantum_tarot/mobile/quantum-tarot-mvp
eas login
```

Enter your Expo credentials. (Don't have an account? Sign up free at expo.dev)

## Step 2: Build APK
```bash
eas build --platform android --profile preview
```

This will:
- Upload your code to Expo's build servers
- Compile the APK (takes ~15 minutes)
- Give you a download link

## Step 3: Download & Install
1. Download the APK from the link Expo provides
2. Transfer to your S25 Ultra (USB, Google Drive, etc.)
3. Open the APK on your phone
4. Tap "Install"
5. Done! 🎉

---

## Alternative: Test Without Building (Instant)

Want to try it right now without building?

```bash
npm start
```

1. Install **Expo Go** app on S25 Ultra (from Play Store)
2. Scan QR code from terminal
3. App runs instantly!

The only difference: Expo Go has a banner at the bottom. The built APK won't have it.

---

## Notes

- **EAS Build is free** for personal projects
- Build time: ~15 minutes
- APK size: ~50-80 MB
- Works on all Android phones (not just S25)
