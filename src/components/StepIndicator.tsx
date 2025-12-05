import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Animated,
} from 'react-native';

const { width, height } = Dimensions.get('window');

interface StepIndicatorProps {
  currentStep: number;
  totalSteps: number;
  completedSteps: number[];
}

const StepIndicator: React.FC<StepIndicatorProps> = ({
  currentStep,
  totalSteps,
  completedSteps,
}) => {
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const steps = [
    { id: 1, title: 'Angle Analysis', icon: '📐', color: '#4ECDC4', desc: 'Measure rib angles and interdistance' },
    { id: 2, title: 'Height Analysis', icon: '📏', color: '#FF6B6B', desc: 'Measure rib heights and lengths' },
  ];

  useEffect(() => {
    initializeAnimations();
  }, []);

  useEffect(() => {
    animateProgress();
  }, [currentStep, completedSteps]);

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
    ]).start();

    // Pulse animation for current step
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
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

  const animateProgress = () => {
    const progress = (completedSteps.length / totalSteps) * 100;
    Animated.timing(progressAnim, {
      toValue: progress,
      duration: 600,
      useNativeDriver: false,
    }).start();
  };

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 100],
    outputRange: ['0%', '100%'],
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
      {/* Progress Bar */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          <Animated.View 
            style={[
              styles.progressFill,
              { width: progressWidth }
            ]} 
          />
        </View>
        <Text style={styles.progressText}>
          {completedSteps.length} of {totalSteps} steps completed
        </Text>
      </View>

      {/* Steps */}
      <View style={styles.stepsContainer}>
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === step.id;
          const isUpcoming = currentStep < step.id;

          return (
            <View key={step.id} style={styles.stepItem}>
              {/* Step Circle */}
              <Animated.View 
                style={[
                  styles.stepCircle,
                  {
                    backgroundColor: isCompleted 
                      ? step.color 
                      : isCurrent 
                        ? step.color 
                        : 'rgba(255, 255, 255, 0.2)',
                    borderColor: isCurrent ? step.color : 'rgba(255, 255, 255, 0.3)',
                    transform: isCurrent ? [{ scale: pulseAnim }] : [{ scale: 1 }],
                  }
                ]}
              >
                {isCompleted ? (
                  <Text style={styles.stepCheckmark}>✓</Text>
                ) : (
                  <Text style={styles.stepIcon}>{step.icon}</Text>
                )}
              </Animated.View>

              {/* Step Info */}
              <View style={styles.stepInfo}>
                <Text style={[
                  styles.stepTitle,
                  {
                    color: isCompleted || isCurrent ? '#FFFFFF' : 'rgba(255, 255, 255, 0.7)',
                    fontWeight: isCurrent ? 'bold' : '600',
                  }
                ]}>
                  {step.title}
                </Text>
                <Text style={[
                  styles.stepDesc,
                  {
                    color: isCompleted || isCurrent ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.5)',
                  }
                ]}>
                  {step.desc}
                </Text>
                {isCurrent && (
                  <View style={styles.currentStepBadge}>
                    <Text style={styles.currentStepText}>Current</Text>
                  </View>
                )}
              </View>

              {/* Connector Line */}
              {index < steps.length - 1 && (
                <View style={[
                  styles.connectorLine,
                  {
                    backgroundColor: isCompleted 
                      ? step.color 
                      : 'rgba(255, 255, 255, 0.2)',
                  }
                ]} />
              )}
            </View>
          );
        })}
      </View>

      {/* Current Step Highlight */}
      {currentStep <= totalSteps && (
        <Animated.View 
          style={[
            styles.currentStepHighlight,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <Text style={styles.highlightIcon}>
            {steps[currentStep - 1]?.icon}
          </Text>
          <Text style={styles.highlightTitle}>
            {steps[currentStep - 1]?.title}
          </Text>
          <Text style={styles.highlightDesc}>
            {steps[currentStep - 1]?.desc}
          </Text>
        </Animated.View>
      )}
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
  progressContainer: {
    marginBottom: 24,
  },
  progressBar: {
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4ECDC4',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    fontWeight: '500',
  },
  stepsContainer: {
    marginBottom: 20,
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 20,
    position: 'relative',
  },
  stepCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    marginRight: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  stepCheckmark: {
    fontSize: 24,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  stepIcon: {
    fontSize: 24,
  },
  stepInfo: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 18,
    marginBottom: 4,
  },
  stepDesc: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  currentStepBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  currentStepText: {
    fontSize: 12,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  connectorLine: {
    position: 'absolute',
    left: 25,
    top: 50,
    width: 2,
    height: 30,
    zIndex: -1,
  },
  currentStepHighlight: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  highlightIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  highlightTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 4,
  },
  highlightDesc: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default StepIndicator; 