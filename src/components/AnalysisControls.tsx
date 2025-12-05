import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import {AnalysisStep} from '../models/AnalysisStep';

const { width, height } = Dimensions.get('window');

const AppColors = {
  primaryBlue: '#1A73E8',
  secondaryBlue: '#4285F4',
  accentBlue: '#8AB4F8',
  backgroundBlue: '#E8F0FE',
  successGreen: '#34A853',
  errorRed: '#EA4335',
  textDark: '#202124',
  textLight: '#5F6368',
};

interface AnalysisControlsProps {
  onAnalyze: () => void;
  onReset: () => void;
  isAnalyzing: boolean;
  isDisabled: boolean;
  currentStep: number;
}

const AnalysisControls: React.FC<AnalysisControlsProps> = ({
  onAnalyze,
  onReset,
  isAnalyzing,
  isDisabled,
  currentStep,
}) => {
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const loadingAnim = useRef(new Animated.Value(0)).current;

  const stepInfo = [
    { step: 1, title: 'Angle Analysis', icon: '📐', color: '#4ECDC4' },
    { step: 2, title: 'Height Analysis', icon: '📏', color: '#FF6B6B' },
  ];

  useEffect(() => {
    initializeAnimations();
  }, []);

  useEffect(() => {
    if (isAnalyzing) {
      animateLoading();
    }
  }, [isAnalyzing]);

  const initializeAnimations = () => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
    ]).start();

    // Pulse animation for action buttons
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const animateLoading = () => {
    Animated.loop(
      Animated.timing(loadingAnim, {
        toValue: 1,
        duration: 1500,
        useNativeDriver: true,
      })
    ).start();
  };

  const loadingRotation = loadingAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <Animated.View 
      style={[
        styles.container,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
        }
      ]}
    >
      {/* Current Step Indicator */}
      <View style={styles.stepIndicator}>
        <View style={styles.stepInfo}>
          <Text style={styles.stepIcon}>{stepInfo[currentStep - 1].icon}</Text>
          <Text style={styles.stepTitle}>{stepInfo[currentStep - 1].title}</Text>
          <Text style={styles.stepNumber}>Step {currentStep}/2</Text>
        </View>
        
        <View style={styles.progressBar}>
          <View 
            style={[
              styles.progressFill,
              { 
                width: `${(currentStep / 2) * 100}%`,
                backgroundColor: stepInfo[currentStep - 1].color
              }
            ]} 
          />
        </View>
      </View>

      {/* Control Buttons */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={[
            styles.resetButton,
            isDisabled && styles.disabledButton
          ]}
          onPress={onReset}
          disabled={isDisabled || isAnalyzing}
          activeOpacity={0.8}
        >
          <Text style={styles.resetButtonIcon}>🔄</Text>
          <Text style={styles.resetButtonText}>Reset</Text>
        </TouchableOpacity>

        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            style={[
              styles.analyzeButton,
              (isDisabled || isAnalyzing) && styles.disabledButton
            ]}
            onPress={onAnalyze}
            disabled={isDisabled || isAnalyzing}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={
                isAnalyzing 
                  ? ['#9E9E9E', '#757575'] 
                  : isDisabled 
                    ? ['#9E9E9E', '#757575']
                    : ['#FF6B6B', '#FF8E53']
              }
              style={styles.analyzeButtonGradient}
            >
              {isAnalyzing ? (
                <Animated.View style={{ transform: [{ rotate: loadingRotation }] }}>
                  <Text style={styles.analyzeButtonIcon}>⏳</Text>
                </Animated.View>
              ) : (
                <Text style={styles.analyzeButtonIcon}>🔍</Text>
              )}
              <Text style={styles.analyzeButtonText}>
                {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>
      </View>

      {/* Status Message */}
      <View style={styles.statusContainer}>
        {isAnalyzing && (
          <View style={styles.statusMessage}>
            <Text style={styles.statusIcon}>⚡</Text>
            <Text style={styles.statusText}>
              Processing {stepInfo[currentStep - 1].title.toLowerCase()}...
            </Text>
          </View>
        )}
        
        {isDisabled && !isAnalyzing && (
          <View style={styles.statusMessage}>
            <Text style={styles.statusIcon}>📋</Text>
            <Text style={styles.statusText}>
              Please capture or select an image first
            </Text>
          </View>
        )}
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  stepIndicator: {
    marginBottom: 20,
  },
  stepInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  stepIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  stepTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    flex: 1,
  },
  stepNumber: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  progressBar: {
    height: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  resetButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 20,
    paddingVertical: 12,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    marginRight: 12,
  },
  resetButtonIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  analyzeButton: {
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    flex: 2,
  },
  disabledButton: {
    opacity: 0.6,
  },
  analyzeButtonGradient: {
    paddingVertical: 16,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  analyzeButtonIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  analyzeButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  statusContainer: {
    alignItems: 'center',
  },
  statusMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  statusIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  statusText: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 14,
    fontWeight: '500',
  },
});

export default AnalysisControls; 