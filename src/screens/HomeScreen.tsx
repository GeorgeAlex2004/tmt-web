import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Animated,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { ApiService } from '../services/ApiService';

const { width, height } = Dimensions.get('window');

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const [isServerAvailable, setIsServerAvailable] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isCheckingServer, setIsCheckingServer] = useState(false);
  
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const loadingRotation = useRef(new Animated.Value(0)).current;
  const serverStatusAnim = useRef(new Animated.Value(0)).current;
  const glowAnim = useRef(new Animated.Value(0)).current;

  const quickStats = [
    { icon: 'analytics', label: 'Analysis Time', value: '< 30s', color: '#4ECDC4' },
    { icon: 'gps-fixed', label: 'Accuracy', value: '±0.5°', color: '#FF6B6B' },
    { icon: 'smartphone', label: 'Mobile Ready', value: 'Any Device', color: '#45B7D1' },
    { icon: 'security', label: 'Secure', value: 'Private Data', color: '#96CEB4' },
  ];

  const analysisSteps = [
    { step: '1', title: 'Select Diameter', desc: 'Choose TMT bar size', icon: 'straighten' },
    { step: '2', title: 'Capture Images', desc: 'Take photos from angles', icon: 'camera-alt' },
    { step: '3', title: 'Adjust Overlay', desc: 'Calibrate measurements', icon: 'tune' },
    { step: '4', title: 'Get Results', desc: 'View analysis & PDF', icon: 'description' },
  ];

  useEffect(() => {
    initializeAnimations();
    checkServerStatus();
    
    // Check server status every 10 seconds
    const serverCheckTimer = setInterval(() => {
      checkServerStatus();
    }, 10000);

    return () => {
      clearInterval(serverCheckTimer);
    };
  }, []);

  const initializeAnimations = () => {
    // Initial animations
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

    // Continuous animations
    Animated.loop(
      Animated.timing(loadingRotation, {
        toValue: 1,
        duration: 1500,
        useNativeDriver: true,
      })
    ).start();

    // Pulse animation for CTA button
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

    // Glow animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(glowAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const checkServerStatus = async () => {
    if (isCheckingServer) return;

    setIsCheckingServer(true);
    try {
      const isAvailable = await ApiService.isServerAvailable();
      setIsServerAvailable(isAvailable);
      
      // Animate server status change
      Animated.timing(serverStatusAnim, {
        toValue: isAvailable ? 1 : 0,
        duration: 500,
        useNativeDriver: true,
      }).start();
    } catch (error) {
      setIsServerAvailable(false);
      Animated.timing(serverStatusAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }).start();
    } finally {
      setIsCheckingServer(false);
      setIsLoading(false);
    }
  };

  const navigateToUserGuidelines = () => {
    navigation.navigate('UserGuidelines');
  };

  const goBackToMainApp = () => {
    navigation.getParent()?.goBack();
  };

  const spin = loadingRotation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const serverStatusScale = serverStatusAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.8, 1],
  });

  const glowOpacity = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 1],
  });

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      {/* Animated Background Elements */}
      <Animated.View style={[styles.backgroundCircle, { opacity: glowOpacity }]} />
      <Animated.View style={[styles.backgroundCircle2, { opacity: glowOpacity }]} />
      
      <ScrollView 
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Section */}
        <Animated.View 
          style={[
            styles.headerContent,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <View style={styles.headerTop}>
            <TouchableOpacity
              style={styles.backButton}
              onPress={goBackToMainApp}
            >
              <Icon name="arrow-back" size={24} color="#00D4FF" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>TMT Analysis Tool</Text>
            <View style={styles.headerSpacer} />
          </View>
        </Animated.View>

        {/* Server Status Section */}
        <Animated.View 
          style={[
            styles.serverStatusSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <Text style={styles.sectionTitle}>System Status</Text>
          
          <View style={styles.serverStatusCard}>
            <View style={styles.serverStatusContent}>
              <View style={styles.serverStatusRow}>
                {isCheckingServer ? (
                  <Animated.View style={{ transform: [{ rotate: spin }] }}>
                    <Icon name="refresh" size={24} color="#00D4FF" />
                  </Animated.View>
                ) : (
                  <Animated.View style={{ transform: [{ scale: serverStatusScale }] }}>
                    <Icon 
                      name={isServerAvailable ? "check-circle" : "error"} 
                      size={24} 
                      color={isServerAvailable ? "#4CAF50" : "#FF6B6B"} 
                    />
                  </Animated.View>
                )}
                <Text style={styles.serverStatusText}>
                  {isCheckingServer ? 'Checking Server...' : 
                   isServerAvailable ? 'Server Online' : 'Server Offline'}
                </Text>
              </View>
              
              <Text style={styles.serverStatusDescription}>
                {isServerAvailable 
                  ? 'AI analysis service is ready for processing'
                  : 'Please check your internet connection and try again'
                }
                  </Text>
            </View>
          </View>
        </Animated.View>

        {/* Quick Stats Section */}
        <View style={styles.statsSection}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
          <View style={styles.statsGrid}>
            {quickStats.map((stat, index) => (
              <View key={index} style={styles.statItem}>
                <View style={[styles.statIconContainer, { backgroundColor: `${stat.color}20` }]}>
                  <Icon name={stat.icon} size={24} color={stat.color} />
                </View>
                <Text style={styles.statValue}>{stat.value}</Text>
                <Text style={styles.statLabel}>{stat.label}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Analysis Steps Section */}
        <View style={styles.stepsSection}>
          <Text style={styles.sectionTitle}>Analysis Process</Text>
          <View style={styles.stepsContainer}>
            {analysisSteps.map((step, index) => (
              <View key={index} style={styles.stepItem}>
                <View style={styles.stepNumber}>
                  <Text style={styles.stepNumberText}>{step.step}</Text>
                </View>
                <View style={styles.stepIconContainer}>
                  <Icon name={step.icon} size={20} color="#00D4FF" />
                </View>
                <Text style={styles.stepTitle}>{step.title}</Text>
                <Text style={styles.stepDescription}>{step.desc}</Text>
              </View>
            ))}
          </View>
            </View>

        {/* CTA Section */}
        <View style={styles.ctaSection}>
          <Animated.View style={[styles.getStartedButton, { transform: [{ scale: pulseAnim }] }]}>
            <TouchableOpacity
              onPress={navigateToUserGuidelines} 
              style={styles.buttonTouchable}
              disabled={!isServerAvailable}
            >
              <LinearGradient
                colors={isServerAvailable ? ['#00D4FF', '#0099CC'] : ['#666666', '#444444']}
                style={styles.buttonGradient}
                start={{x: 0, y: 0}}
                end={{x: 1, y: 1}}
              >
                <Icon name="camera-alt" size={24} color="white" style={styles.buttonIcon} />
                <Text style={styles.getStartedButtonText}>
                  {isServerAvailable ? 'Start Analysis' : 'Server Offline'}
                </Text>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
          
          <Text style={styles.ctaSubtitle}>
            {isServerAvailable 
              ? 'Click to begin your TMT bar analysis'
              : 'Please wait for server to come online'
            }
          </Text>
          </View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  backgroundCircle: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    top: 20,
    right: -50,
  },
  backgroundCircle2: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: 'rgba(0, 212, 255, 0.05)',
    bottom: 20,
    left: -30,
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
  },
  headerContent: {
    marginBottom: 30,
  },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  backButton: {
    position: 'absolute',
    top: 20,
    left: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
    zIndex: 10,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    flex: 1,
    textAlign: 'center',
  },
  headerSpacer: {
    width: 40,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 15,
    textAlign: 'center',
  },
  serverStatusSection: {
    marginBottom: 30,
  },
  serverStatusCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  serverStatusContent: {
    alignItems: 'center',
  },
  serverStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  serverStatusText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginLeft: 10,
  },
  serverStatusDescription: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 20,
  },
  statsSection: {
    marginBottom: 30,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statItem: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  statIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  stepsSection: {
    marginBottom: 30,
  },
  stepsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  stepItem: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  stepNumber: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 212, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  stepNumberText: {
    color: '#00D4FF',
    fontWeight: 'bold',
    fontSize: 16,
  },
  stepIconContainer: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  stepDescription: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 16,
  },
  ctaSection: {
    alignItems: 'center',
    marginBottom: 40,
  },
  getStartedButton: {
    borderRadius: 30,
    overflow: 'hidden',
    marginBottom: 15,
    elevation: 8,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonTouchable: {
    borderRadius: 30,
    overflow: 'hidden',
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 18,
    paddingHorizontal: 40,
  },
  buttonIcon: {
    marginRight: 10,
  },
  getStartedButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  ctaSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default HomeScreen; 