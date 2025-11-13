# 📱 iOS App Store Deployment Guide - LunatiQ

Complete guide for building and submitting LunatiQ to the Apple App Store.

## ✅ iOS Compatibility Checklist

Your app is **100% iOS compatible** with the following features:

- ✅ **Expo SDK 54** - Latest stable Expo version
- ✅ **React Native 0.81** - Latest RN version
- ✅ **iOS 13.0+ deployment target** - Covers 99%+ of active devices
- ✅ **iPad support** - Universal app works on iPhone and iPad
- ✅ **Platform-specific fonts** - Uses iOS-native fonts (Courier)
- ✅ **Dark mode** - Supports iOS dark mode
- ✅ **Safe Area** - Respects notches and home indicators
- ✅ **No camera/microphone** - No privacy-sensitive permissions required
- ✅ **Offline-first** - Works without internet connection
- ✅ **No encryption export issues** - Properly declared in config

---

## 📋 Prerequisites

### 1. Apple Developer Account
- **Cost:** $99/year
- **Sign up:** https://developer.apple.com/programs/
- **Required for:** App Store submission

### 2. Install EAS CLI
```bash
npm install -g eas-cli
```

### 3. Login to Expo
```bash
eas login
```

### 4. Configure EAS Project
```bash
cd quantum_tarot/mobile/quantum-tarot-mvp
eas init
```

This will:
- Create an Expo project ID
- Update `app.json` with your project ID
- Link your project to Expo servers

---

## 🏗️ Building for iOS

### Step 1: Install Dependencies
```bash
npm install
```

### Step 2: Build for iOS (First Time)
```bash
eas build --platform ios --profile production
```

**What happens:**
1. EAS will ask if you want to generate credentials automatically → **Say YES**
2. Apple Push Notification service key will be generated
3. Distribution certificate will be created
4. Provisioning profile will be created
5. Build will run on Expo's servers (takes 10-20 minutes)

**Result:** You'll get a `.ipa` file (iOS App Store package)

### Step 3: Download the IPA
```bash
# The CLI will give you a download link
# Or visit: https://expo.dev/accounts/YOUR_ACCOUNT/projects/YOUR_PROJECT/builds
```

---

## 📲 Testing on Physical iPhone/iPad

### Option 1: TestFlight (Recommended)
TestFlight is Apple's official beta testing platform.

```bash
# Build and automatically submit to TestFlight
eas build --platform ios --profile production --auto-submit
```

**Steps:**
1. Build completes and uploads to App Store Connect
2. Apple processes the build (5-10 minutes)
3. Open TestFlight app on your iPhone
4. Install LunatiQ
5. Test all features!

### Option 2: Development Build
```bash
# For testing during development
eas build --platform ios --profile development
```

Then install using:
- Expo Go app
- USB connection with Xcode
- Internal distribution

---

## 🍎 App Store Connect Setup

### 1. Create App Listing
1. Go to https://appstoreconnect.apple.com
2. Click **"My Apps"** → **"+"** → **"New App"**
3. Fill in:
   - **Platform:** iOS
   - **Name:** LunatiQ
   - **Primary Language:** English (U.S.)
   - **Bundle ID:** `com.aphoticshaman.lunatiq`
   - **SKU:** `lunatiq-001` (unique identifier)

### 2. App Information
- **Category:** Lifestyle > Entertainment
- **Secondary Category:** Health & Fitness (optional)
- **Content Rights:** Check if you own all content

### 3. Pricing & Availability
- **Price:** Free (or set your price)
- **Availability:** All countries (or select specific ones)

### 4. Prepare App Metadata

#### App Name & Subtitle
- **Name:** LunatiQ
- **Subtitle:** Cyberpunk Tarot with AI Interpretation (max 30 characters)

#### Description (4000 character max)
```
Experience tarot like never before with LunatiQ - a cyberpunk-themed tarot reading app powered by advanced AI interpretation.

FEATURES:
• 78 complete tarot deck with rich symbolism
• Multiple spread types (3-card, Celtic Cross, etc.)
• AI-powered interpretations using psychology principles
• Beautiful cyberpunk neon aesthetic with Matrix rain effects
• Offline-first - works without internet
• Private and secure - all readings stored locally
• Multiple interpretation voices (analytical, intuitive, supportive, etc.)

QUANTUM RANDOMIZATION:
Uses hardware-based random number generation for truly random card draws, combined with Fisher-Yates shuffling algorithm for maximum entropy.

INTERPRETATION ENGINE:
Our AI combines:
- Jungian archetypes and shadow work
- Cognitive Behavioral Therapy (CBT) principles
- Dialectical Behavior Therapy (DBT) concepts  
- Army Master Resilience Training (MRT) techniques
- Traditional tarot symbolism and numerology

PRIVACY:
• No account required
• No data collection
• No ads
• No internet connection needed
• All readings stored locally on your device

Perfect for:
- Tarot enthusiasts
- Personal development
- Daily reflection and journaling
- Self-awareness and mindfulness
- Decision-making support

LunatiQ is not a fortune-telling app - it's a tool for self-reflection, personal growth, and gaining perspective on life's challenges.
```

#### Keywords (100 character max, comma-separated)
```
tarot,divination,cards,oracle,ai,cyberpunk,mindfulness,therapy,psychology,self-help
```

#### Screenshots (Required)
You need screenshots for different device sizes:
- **6.5" iPhone** (1242 x 2688 px or 1290 x 2796 px)
- **5.5" iPhone** (1242 x 2208 px)
- **iPad Pro 12.9"** (2048 x 2732 px)

**How to capture:**
1. Run app on iOS simulator
2. Take screenshots showing:
   - Home screen with rainbow LunatiQ logo
   - Card drawing screen with Matrix rain
   - Card spread layout
   - AI interpretation screen
   - Settings screen

### 5. App Review Information

#### Contact Information
- **First Name:** [Your Name]
- **Last Name:** [Your Last Name]
- **Phone Number:** [Your Phone]
- **Email:** [Your Email]

#### Review Notes
```
LunatiQ is a tarot card reading app with AI-powered interpretations.

TEST ACCOUNT: No account required

HOW TO TEST:
1. Tap "Start Reading"
2. Select a spread type (e.g., "3-Card Spread")
3. Set an intention (optional)
4. Watch the card drawing animation
5. View the AI-generated interpretation

The app uses hardware random number generation for card shuffling. All data is stored locally using AsyncStorage. No internet connection is required.

AI interpretations are generated using psychological principles (CBT, DBT, Jungian psychology) combined with traditional tarot symbolism. This is a tool for self-reflection, not fortune-telling.

No in-app purchases, ads, or data collection.
```

#### Age Rating
- **Age Rating:** 12+
- **Simulated Gambling:** No
- **Contests:** No
- **Unrestricted Web Access:** No
- **Horror/Fear Themes:** No

### 6. Upload Build
```bash
# Submit to App Store Connect
eas submit --platform ios
```

Or manually:
1. Download the `.ipa` file from EAS
2. Use Transporter app (Mac App Store) to upload
3. Wait 5-10 minutes for processing

---

## 🚀 Submitting for Review

### 1. Select Build
In App Store Connect:
1. Go to your app
2. Click **"App Store"** tab
3. Click **"+"** next to **"iOS App"**
4. Select your uploaded build

### 2. Submit for Review
1. Click **"Submit for Review"**
2. Answer questionnaire:
   - **Export Compliance:** No (already declared in app.json)
   - **Advertising Identifier:** No
   - **Content Rights:** Yes, you own all rights

### 3. Review Time
- **Initial Review:** 24-48 hours typically
- **Updates:** Usually faster (12-24 hours)

---

## 📊 Apple App Store Requirements (2025-2026)

### Current Requirements (2025)
- ✅ **iOS 13.0+** deployment target (you have this)
- ✅ **Xcode 15/16** compatible (Expo handles this)
- ✅ **iOS 18 SDK** support (Expo SDK 54 uses latest)

### Future Requirements (April 2026)
Apple hasn't announced "Xcode 26" or "iOS 26" yet. The requirements you saw likely refer to:
- **iOS 19 SDK** (expected late 2025)
- **Xcode 17** (expected late 2025)

**Don't worry:** Expo automatically updates to support the latest iOS versions. When iOS 19 is released:
1. Update to latest Expo SDK: `expo upgrade`
2. Rebuild: `eas build --platform ios`
3. That's it!

---

## 🔍 Common Rejection Reasons & Solutions

### 1. Missing Privacy Policy
**Solution:** Add privacy policy to app.json:
```json
"ios": {
  "privacyManifests": {
    "NSPrivacyAccessedAPITypes": []
  }
}
```

### 2. App Name Already Taken
**Solution:** Use "LunatiQ - Tarot Reader" or similar variant

### 3. Gambling Concerns
**Solution:** In review notes, clarify: "This is a self-reflection tool, not gambling or fortune-telling for money"

### 4. 2.5.2 - Accurate Metadata
**Solution:** Don't claim the app can "predict the future" - focus on "self-reflection" and "personal insight"

### 5. Missing In-App Purchase Disclosure
**Solution:** If you add IAP later, properly implement StoreKit

---

## 🔄 Updating Your App

### Increment Version
Update `app.json`:
```json
{
  "version": "1.0.1",
  "ios": {
    "buildNumber": "2"
  }
}
```

### Build New Version
```bash
eas build --platform ios --profile production
```

### Submit Update
```bash
eas submit --platform ios
```

---

## 🛠️ Troubleshooting

### Build Fails
```bash
# Clear cache and retry
eas build --platform ios --clear-cache
```

### Can't Sign In to Expo
```bash
# Logout and login again
eas logout
eas login
```

### TestFlight Not Showing Build
- Wait 10-15 minutes after upload
- Check App Store Connect for processing status
- Ensure you added yourself as internal tester

### "Missing Compliance" Warning
Already handled in `app.json`:
```json
"config": {
  "usesNonExemptEncryption": false
}
```

---

## 📞 Support Resources

- **Expo Docs:** https://docs.expo.dev/
- **EAS Build:** https://docs.expo.dev/build/introduction/
- **App Store Connect:** https://appstoreconnect.apple.com
- **Apple Developer:** https://developer.apple.com/support/
- **App Store Review Guidelines:** https://developer.apple.com/app-store/review/guidelines/

---

## ✨ Quick Command Reference

```bash
# Install dependencies
npm install

# Login to Expo
eas login

# Initialize EAS
eas init

# Build for iOS
eas build --platform ios

# Submit to App Store
eas submit --platform ios

# Check build status
eas build:list

# Update Expo SDK
expo upgrade

# Run on iOS simulator locally
npm run ios
```

---

## 🎉 You're Ready!

Your LunatiQ app is fully configured for iOS deployment. The codebase is 100% iOS compatible with:
- ✅ Proper platform-specific code
- ✅ iOS-native fonts and UI
- ✅ Safe area handling for notches
- ✅ Dark mode support
- ✅ iPad compatibility
- ✅ No privacy-sensitive permissions
- ✅ Offline functionality

Just follow the steps above to submit to the App Store!

**Good luck! 🚀**
