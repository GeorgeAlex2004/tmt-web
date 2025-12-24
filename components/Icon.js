import React from 'react';
import { Platform } from 'react-native';

// Use @expo/vector-icons which works on both web and native
// This is the recommended approach for Expo apps
let IconComponent;

try {
  // Primary: Use @expo/vector-icons (works on web and native)
  const { MaterialIcons } = require('@expo/vector-icons');
  IconComponent = MaterialIcons;
} catch (e) {
  // Fallback 1: Try react-native-vector-icons (native only)
  try {
    IconComponent = require('react-native-vector-icons/MaterialIcons').default;
  } catch (e2) {
    // Fallback 2: Create a simple text-based icon for web
    IconComponent = ({ name, size, color, style, ...props }) => {
      // Map common Material Icons to Unicode/Emoji fallbacks
      const iconMap = {
        'home': '🏠',
        'settings': '⚙️',
        'camera-alt': '📷',
        'photo-camera': '📷',
        'image': '🖼️',
        'history': '📜',
        'logout': '🚪',
        'arrow-back': '←',
        'arrow-forward': '→',
        'check': '✓',
        'close': '✕',
        'menu': '☰',
        'search': '🔍',
        'analytics': '📊',
        'gps-fixed': '📍',
        'smartphone': '📱',
        'security': '🔒',
        'straighten': '📏',
        'tune': '🎛️',
        'description': '📄',
        'precision-manufacturing': '⚙️',
        'refresh': '🔄',
        'download': '⬇️',
        'share': '📤',
        'print': '🖨️',
        'edit': '✏️',
        'delete': '🗑️',
        'delete-forever': '🗑️',
        'add': '+',
        'remove': '−',
        'info': 'ℹ️',
        'warning': '⚠️',
        'error': '❌',
        'check-circle': '✓',
        'cancel': '✕',
        'lightbulb-outline': '💡',
        'blur-off': '🔍',
        'crop-free': '📐',
        'filter-none': '🖼️',
        'view-module': '📊',
        'report-problem': '⚠️',
        'stay-primary-portrait': '📱',
        'science': '🔬',
      };
      
      const iconChar = iconMap[name] || '?';
      
      if (Platform.OS === 'web') {
        return (
          <span
            style={{
              fontSize: size || 24,
              color: color || '#000',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              ...style,
            }}
            {...props}
          >
            {iconChar}
          </span>
        );
      }
      
      // For native, return a simple View with text
      const { View, Text } = require('react-native');
      return (
        <View style={[{ alignItems: 'center', justifyContent: 'center' }, style]} {...props}>
          <Text style={{ fontSize: size || 24, color: color || '#000' }}>{iconChar}</Text>
        </View>
      );
    };
  }
}

export default IconComponent;

