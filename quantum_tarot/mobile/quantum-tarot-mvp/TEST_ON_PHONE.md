# 🚀 Test on Your S25 Ultra - 2 Minutes

Your laptop just needs to finish installing Windows, then:

## Step 1: Open Terminal/Command Prompt

Navigate to the project:
```bash
cd quantum_tarot/mobile/quantum-tarot-mvp
```

## Step 2: Start Dev Server
```bash
npm start
```

You'll see:
- A QR code in the terminal
- A URL like `exp://192.168.x.x:8081`

## Step 3: On Your S25 Ultra

1. **Install Expo Go** from Play Store (if not already installed)
2. **Open Expo Go** app
3. **Scan the QR code** from your terminal

That's it! The app will load in seconds.

---

## ✨ What You'll See

1. **Welcome screen** with ASCII art logo
2. **Onboarding** - Enter your name & birthday
3. **Choose reading type** (Career, Romance, Wellness, etc.)
4. **10 personality questions** with progress bar
5. **Set your intention** and choose spread
6. **Quantum card drawing** with animation
7. **Your reading** with ASCII cards and personalized interpretation
8. **Settings** to change color themes

---

## 🎨 Try All 5 Themes!

Go to Settings and switch between:
- **Matrix Green** - Classic hacker vibes
- **Amber Terminal** - Retro 80s terminal
- **Cyan Retro** - Cool blue aesthetic
- **Vaporwave** - Pink/purple nostalgia
- **White Classic** - Clean monochrome

---

## 💡 Troubleshooting

**Can't scan QR code?**
- Make sure laptop and phone are on same WiFi network
- Or manually type the `exp://` URL shown in terminal into Expo Go

**"Something went wrong"?**
```bash
# Clear cache and restart
npm start --clear
```

**Need to stop the server?**
- Press `Ctrl+C` in terminal

---

## 🎯 Next Step: Build APK

Once you test it and love it, build the standalone APK:
```bash
eas login
eas build --platform android --profile preview
```

Takes ~15 minutes, then you get a download link for the APK!
