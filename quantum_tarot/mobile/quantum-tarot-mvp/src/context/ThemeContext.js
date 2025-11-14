/**
 * THEME CONTEXT - Cyberpunk theme provider (future-proofing)
 */

import React, { createContext, useContext, useState } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  // Currently only cyberpunk theme, but structure for future customization
  const [currentTheme] = useState({
    name: 'cyberpunk',
    // Flat structure for easy access
    primary: '#00FFFF',
    secondary: '#FF00FF',
    accent: '#FFFF00',
    background: '#000000',
    text: '#FFFFFF',
    textDim: '#CCCCCC',
    border: '#00FFFF',
  });

  return (
    <ThemeContext.Provider value={{ theme: currentTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}
