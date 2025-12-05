import React from 'react';
import { View, Platform, StyleSheet } from 'react-native';

const WebLinearGradient = ({ colors, style, children, start, end }) => {
  if (Platform.OS === 'web') {
    const gradientColors = colors.join(', ');
    const startX = (start?.x || 0) * 100;
    const startY = (start?.y || 0) * 100;
    const endX = (end?.x || 1) * 100;
    const endY = (end?.y || 0) * 100;

    return (
      <View
        style={[
          style,
          {
            background: `linear-gradient(to right, ${gradientColors})`,
            backgroundImage: `linear-gradient(${startX}deg, ${gradientColors})`,
          },
        ]}
      >
        {children}
      </View>
    );
  }

  // For native platforms, use a simple colored View as fallback
  return (
    <View
      style={[
        style,
        {
          backgroundColor: colors[0] || 'transparent',
        },
      ]}
    >
      {children}
    </View>
  );
};

export default WebLinearGradient; 