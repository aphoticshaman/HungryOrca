/**
 * i18n (Internationalization) Setup
 * Provides translation support for the app
 */

import en from './en.json';

// Current language (can be extended to support more languages)
let currentLanguage = 'en';

const translations = {
  en
};

/**
 * Get translated string by key path
 * @param {string} key - Dot-notation path to translation (e.g., "common.next")
 * @param {object} params - Optional parameters to interpolate
 * @returns {string} - Translated string
 */
export function t(key, params = {}) {
  const keys = key.split('.');
  let value = translations[currentLanguage];

  for (const k of keys) {
    if (value && typeof value === 'object') {
      value = value[k];
    } else {
      console.warn(`Translation missing for key: ${key}`);
      return key;
    }
  }

  if (typeof value !== 'string') {
    console.warn(`Translation is not a string for key: ${key}`);
    return key;
  }

  // Replace interpolation parameters {{param}}
  return value.replace(/\{\{(\w+)\}\}/g, (match, param) => {
    return params[param] !== undefined ? params[param] : match;
  });
}

/**
 * Set current language
 * @param {string} lang - Language code (e.g., 'en', 'es')
 */
export function setLanguage(lang) {
  if (translations[lang]) {
    currentLanguage = lang;
  } else {
    console.warn(`Language '${lang}' not supported, staying with '${currentLanguage}'`);
  }
}

/**
 * Get current language
 * @returns {string} - Current language code
 */
export function getCurrentLanguage() {
  return currentLanguage;
}

/**
 * Get all available languages
 * @returns {string[]} - Array of supported language codes
 */
export function getAvailableLanguages() {
  return Object.keys(translations);
}

// Export default t function for convenience
export default t;
