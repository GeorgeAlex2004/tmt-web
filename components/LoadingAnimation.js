import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Dimensions,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const LoadingAnimation = ({ visible, title = "Processing Image...", subtitle = "SAM model processing may take 2-3 minutes" }) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  
  const [currentStep, setCurrentStep] = useState(0);

  const progressSteps = [
    "Loading SAM model...",
    "Analyzing image...",
    "Segmenting TMT bar...",
    "Calculating measurements...",
    "Generating results...",
  ];

  useEffect(() => {
    if (visible) {
      // Start animations
      Animated.parallel([
        // Pulse animation
        Animated.loop(
          Animated.sequence([
            Animated.timing(pulseAnim, {
              toValue: 1.2,
              duration: 1000,
              useNativeDriver: true,
            }),
            Animated.timing(pulseAnim, {
              toValue: 1,
              duration: 1000,
              useNativeDriver: true,
            }),
          ])
        ),
        // Rotation animation
        Animated.loop(
          Animated.timing(rotateAnim, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          })
        ),
        // Fade in
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
        // Scale in
        Animated.spring(scaleAnim, {
          toValue: 1,
          tension: 50,
          friction: 7,
          useNativeDriver: true,
        }),
      ]).start();

      // Start progress animation
      Animated.timing(progressAnim, {
        toValue: 1,
        duration: 30000, // 30 seconds total
        useNativeDriver: false,
      }).start();

      // Simulate progress steps
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev < progressSteps.length - 1) {
            return prev + 1;
          }
          return prev;
        });
      }, 6000); // Change step every 6 seconds

      return () => clearInterval(stepInterval);
    } else {
      // Reset animations
      pulseAnim.setValue(1);
      rotateAnim.setValue(0);
      fadeAnim.setValue(0);
      scaleAnim.setValue(0.8);
      progressAnim.setValue(0);
      setCurrentStep(0);
    }
  }, [visible]);

  const spin = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <Modal
      transparent={true}
      animationType="none"
      visible={visible}
      onRequestClose={() => {}}
    >
      <Animated.View 
        style={[
          styles.modalOverlay,
          {
            opacity: fadeAnim,
          }
        ]}
      >
        <Animated.View 
          style={[
            styles.modalContent,
            {
              transform: [
                { scale: scaleAnim },
              ],
            }
          ]}
        >
          <LinearGradient
            colors={['#0D47A1', '#1976D2', '#42A5F5']}
            style={styles.gradientBackground}
          >
            {/* Animated Icon */}
            <Animated.View
              style={[
                styles.iconContainer,
                {
                  transform: [
                    { scale: pulseAnim },
                    { rotate: spin },
                  ],
                },
              ]}
            >
              <Icon name="camera-alt" size={40} color="white" />
            </Animated.View>

            {/* Title */}
            <Text style={styles.title}>{title}</Text>

            {/* Progress Bar */}
            <View style={styles.progressBarContainer}>
              <View style={styles.progressBarBackground}>
                <Animated.View 
                  style={[
                    styles.progressBarFill,
                    {
                      width: progressWidth,
                    }
                  ]}
                />
              </View>
              <Text style={styles.progressText}>
                {Math.round(progressAnim._value * 100)}%
              </Text>
            </View>

            {/* Current Step */}
            <View style={styles.currentStepContainer}>
              <Animated.View
                style={[
                  styles.stepIndicator,
                  {
                    opacity: fadeAnim,
                  },
                ]}
              >
                <Icon name="check-circle" size={20} color="#4CAF50" />
                <Text style={styles.currentStepText}>
                  {progressSteps[currentStep]}
                </Text>
              </Animated.View>
            </View>

            {/* Progress Steps */}
            <View style={styles.progressContainer}>
              {progressSteps.map((step, index) => (
                <Animated.View
                  key={index}
                  style={[
                    styles.progressStep,
                    {
                      opacity: fadeAnim.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0, index <= currentStep ? 1 : 0.5],
                      }),
                    },
                  ]}
                >
                  <View style={[
                    styles.stepDot,
                    index <= currentStep && styles.stepDotActive
                  ]} />
                  <Text style={[
                    styles.stepText,
                    index <= currentStep && styles.stepTextActive
                  ]}>
                    {step}
                  </Text>
                </Animated.View>
              ))}
            </View>

            {/* Subtitle */}
            <Text style={styles.subtitle}>{subtitle}</Text>
            <Text style={styles.subtitle}>Please be patient</Text>

            {/* Animated Dots */}
            <View style={styles.dotsContainer}>
              {[0, 1, 2].map((index) => (
                <Animated.View
                  key={index}
                  style={[
                    styles.dot,
                    {
                      transform: [
                        {
                          scale: pulseAnim.interpolate({
                            inputRange: [1, 1.2],
                            outputRange: [1, 1.3],
                          }),
                        },
                      ],
                      opacity: pulseAnim.interpolate({
                        inputRange: [1, 1.2],
                        outputRange: [0.6, 1],
                      }),
                    },
                  ]}
                />
              ))}
            </View>
          </LinearGradient>
        </Animated.View>
      </Animated.View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: width * 0.85,
    maxWidth: 350,
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 15,
  },
  gradientBackground: {
    padding: 40,
    alignItems: 'center',
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 20,
  },
  progressBarContainer: {
    width: '100%',
    marginBottom: 20,
    alignItems: 'center',
  },
  progressBarBackground: {
    width: '100%',
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#4CAF50',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: 'white',
    marginTop: 8,
    fontWeight: 'bold',
  },
  currentStepContainer: {
    width: '100%',
    marginBottom: 15,
  },
  stepIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: 10,
    borderRadius: 10,
  },
  currentStepText: {
    fontSize: 14,
    color: 'white',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  progressContainer: {
    width: '100%',
    marginBottom: 20,
  },
  progressStep: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    marginRight: 12,
  },
  stepDotActive: {
    backgroundColor: '#4CAF50',
  },
  stepText: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    flex: 1,
  },
  stepTextActive: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginBottom: 5,
  },
  dotsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    marginHorizontal: 4,
  },
});

export default LoadingAnimation; 