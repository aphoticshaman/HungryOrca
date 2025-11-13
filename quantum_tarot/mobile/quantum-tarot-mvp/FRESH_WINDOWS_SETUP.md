# 🪟 Fresh Windows Setup Guide

Your Windows just finished installing. Here's what you need to do:

---

## Step 1: Install Node.js (5 minutes)

1. Go to https://nodejs.org/
2. Download the **LTS version** (left button - should be v20.x or v18.x)
3. Run the installer
4. Click "Next" through everything (default options are fine)
5. Restart your computer

**Verify it worked:**
Open Command Prompt and type:
```cmd
node --version
npm --version
```

Should show version numbers.

---

## Step 2: Install Git (3 minutes)

1. Go to https://git-scm.com/download/win
2. Download and run installer
3. Click "Next" through everything (defaults are fine)

**Verify it worked:**
```cmd
git --version
```

---

## Step 3: Clone Your Repository (2 minutes)

Open Command Prompt or PowerShell:

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/aphoticshaman/HungryOrca.git
cd HungryOrca\quantum_tarot\mobile\quantum-tarot-mvp
```

---

## Step 4: Install Project Dependencies (3 minutes)

```cmd
npm install
```

This downloads all the packages your app needs. Takes a few minutes.

---

## Step 5: Start Development Server (1 minute)

```cmd
npm start
```

You'll see:
- A QR code
- A URL
- "Metro waiting on exp://..."

**Keep this window open!**

---

## Step 6: On Your S25 Ultra

1. **Install Expo Go** from Play Store (if not already)
2. **Open Expo Go**
3. **Scan the QR code** from your laptop screen

App loads in 10-20 seconds!

---

## 🎯 Full Command Sequence (Copy/Paste)

Once Node.js and Git are installed:

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/aphoticshaman/HungryOrca.git
cd HungryOrca\quantum_tarot\mobile\quantum-tarot-mvp
npm install
npm start
```

Then scan QR with Expo Go on your phone.

---

## 🐛 Troubleshooting

**"npm is not recognized"**
- Restart computer after installing Node.js
- Or close and reopen Command Prompt

**"git is not recognized"**
- Restart computer after installing Git
- Or close and reopen Command Prompt

**QR code doesn't scan?**
- Make sure laptop and phone are on **same WiFi**
- Or manually type the `exp://` URL into Expo Go

**"Something went wrong" in Expo Go?**
- Press `r` in the Command Prompt to reload
- Or stop (Ctrl+C) and run `npm start` again

**Dependencies taking forever?**
- Normal on first install! Can take 5-10 minutes
- Just let it run

---

## ⏱️ Total Time: ~20 minutes

- Install Node.js: 5 min
- Install Git: 3 min
- Clone repo: 2 min
- npm install: 5 min
- npm start: 1 min
- Load on phone: 1 min

Then you're testing your quantum tarot app! 🔮

---

## 🎉 After Testing

If you love it and want the standalone APK:

```cmd
npm install -g eas-cli
eas login
eas build --platform android --profile preview
```

Takes ~15 minutes, then download and install the APK on your phone.
