/**
 * Generate placeholder images for the app
 * Run: node generate-placeholders.js
 */

const fs = require('fs');
const path = require('path');

// Create SVG images then convert to PNG if needed
const createIconSVG = (size = 1024) => `
<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0A0E27;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1a1f3a;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="${size}" height="${size}" fill="url(#bg)"/>
  <circle cx="${size/2}" cy="${size/2}" r="${size/3}" fill="none" stroke="#00D9FF" stroke-width="8" opacity="0.6"/>
  <circle cx="${size/2}" cy="${size/2}" r="${size/4}" fill="none" stroke="#FF00FF" stroke-width="6" opacity="0.4"/>
  <text x="50%" y="55%" font-family="Arial, sans-serif" font-size="${size/8}" fill="#00D9FF" text-anchor="middle" font-weight="bold">🔮</text>
  <text x="50%" y="75%" font-family="Arial, sans-serif" font-size="${size/20}" fill="white" text-anchor="middle" opacity="0.8">QT</text>
</svg>`;

const createSplashSVG = () => `
<svg width="1242" height="2688" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="splash-bg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0A0E27;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#1a1f3a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0A0E27;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="1242" height="2688" fill="url(#splash-bg)"/>
  <circle cx="621" cy="1100" r="300" fill="none" stroke="#00D9FF" stroke-width="4" opacity="0.3"/>
  <circle cx="621" cy="1100" r="250" fill="none" stroke="#FF00FF" stroke-width="3" opacity="0.2"/>
  <text x="50%" y="40%" font-family="Arial, sans-serif" font-size="200" fill="#00D9FF" text-anchor="middle">🔮</text>
  <text x="50%" y="52%" font-family="Arial, sans-serif" font-size="80" fill="white" text-anchor="middle" font-weight="bold">Quantum Tarot</text>
  <text x="50%" y="56%" font-family="Arial, sans-serif" font-size="40" fill="#00D9FF" text-anchor="middle" opacity="0.7">Unlock Your Future</text>
</svg>`;

// Ensure assets directory exists
const assetsDir = path.join(__dirname, 'assets');
if (!fs.existsSync(assetsDir)) {
  fs.mkdirSync(assetsDir, { recursive: true });
}

// Write SVG files
fs.writeFileSync(path.join(assetsDir, 'icon.svg'), createIconSVG(1024));
fs.writeFileSync(path.join(assetsDir, 'adaptive-icon.svg'), createIconSVG(1024));
fs.writeFileSync(path.join(assetsDir, 'splash.svg'), createSplashSVG());

console.log('✅ SVG placeholder images created!');
console.log('\nCreated files:');
console.log('  - assets/icon.svg');
console.log('  - assets/adaptive-icon.svg');
console.log('  - assets/splash.svg');
console.log('\n📝 To convert SVGs to PNG, you can:');
console.log('  1. Use an online converter like cloudconvert.com');
console.log('  2. Use Figma/Photoshop to export');
console.log('  3. Use ImageMagick: convert icon.svg icon.png');
console.log('  4. Or just use the SVGs directly (React Native supports SVG)');
console.log('\n💡 For production, replace with custom designed icons!');
