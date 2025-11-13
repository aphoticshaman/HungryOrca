import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ASCII Color Themes
export const ASCII_THEMES = {
  matrix_green: {
    id: 'matrix_green',
    name: 'Matrix Green',
    text: '#00FF41',
    textDim: '#00AA2B',
    background: '#000000',
    border: '#00FF41',
    accent: '#00FF41'
  },
  amber_terminal: {
    id: 'amber_terminal',
    name: 'Amber Terminal',
    text: '#FFB000',
    textDim: '#CC8800',
    background: '#0C0C0C',
    border: '#FFB000',
    accent: '#FFC020'
  },
  cyan_retro: {
    id: 'cyan_retro',
    name: 'Cyan Retro',
    text: '#00FFFF',
    textDim: '#00AAAA',
    background: '#000033',
    border: '#00FFFF',
    accent: '#00FFFF'
  },
  vaporwave: {
    id: 'vaporwave',
    name: 'Vaporwave',
    text: '#FF6AD5',
    textDim: '#CC55AA',
    background: '#2A1B3D',
    border: '#00F0FF',
    accent: '#FF6AD5'
  },
  white_classic: {
    id: 'white_classic',
    name: 'White Classic',
    text: '#FFFFFF',
    textDim: '#AAAAAA',
    background: '#000000',
    border: '#FFFFFF',
    accent: '#FFFFFF'
  }
};

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [currentTheme, setCurrentTheme] = useState(ASCII_THEMES.matrix_green);

  useEffect(() => {
    loadTheme();
  }, []);

  async function loadTheme() {
    try {
      const savedTheme = await AsyncStorage.getItem('asciiTheme');
      if (savedTheme && ASCII_THEMES[savedTheme]) {
        setCurrentTheme(ASCII_THEMES[savedTheme]);
      }
    } catch (error) {
      console.error('Failed to load theme:', error);
    }
  }

  async function changeTheme(themeId) {
    if (ASCII_THEMES[themeId]) {
      setCurrentTheme(ASCII_THEMES[themeId]);
      try {
        await AsyncStorage.setItem('asciiTheme', themeId);
      } catch (error) {
        console.error('Failed to save theme:', error);
      }
    }
  }

  return (
    <ThemeContext.Provider value={{ theme: currentTheme, changeTheme, themes: ASCII_THEMES }}>
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
