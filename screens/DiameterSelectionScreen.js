import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Animated,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const DiameterSelectionScreen = ({ navigation }) => {
  const [selectedDiameter, setSelectedDiameter] = useState(null);
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const glowAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    // Start entrance animations
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();

    // Start glow animation
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

    // Start pulse animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.02,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const diameters = [
    {
      value: 8,
      label: '8 mm',
      rimMin: 0.56,
      rimMax: 0.8,
      icon: 'filter-1',
      color: ['#00D4FF', '#0099CC'],
      glowColor: 'rgba(0, 212, 255, 0.3)',
    },
    {
      value: 10,
      label: '10 mm',
      rimMin: 0.7,
      rimMax: 1.0,
      icon: 'filter-2',
      color: ['#45B7D1', '#2196F3'],
      glowColor: 'rgba(69, 183, 209, 0.3)',
    },
    { 
      value: 12, 
      label: '12 mm',
      rimMin: 0.84,
      rimMax: 1.2,
      icon: 'filter-3',
      color: ['#00C853', '#B2FF59'],
      glowColor: 'rgba(0, 200, 83, 0.3)',
    },
    { 
      value: 16, 
      label: '16 mm',
      rimMin: 1.12,
      rimMax: 1.6,
      icon: 'filter-4',
      color: ['#FF6B6B', '#FF5252'],
      glowColor: 'rgba(255, 107, 107, 0.3)',
    },
    { 
      value: 20, 
      label: '20 mm',
      rimMin: 1.4,
      rimMax: 2.0,
      icon: 'filter-5',
      color: ['#4ECDC4', '#26A69A'],
      glowColor: 'rgba(78, 205, 196, 0.3)',
    },
    { 
      value: 25, 
      label: '25 mm',
      rimMin: 1.75,
      rimMax: 2.5,
      icon: 'filter-6',
      color: ['#A78BFA', '#8B5CF6'],
      glowColor: 'rgba(167, 139, 250, 0.3)',
    },
  ];

  const handleDiameterSelect = (diameter) => {
    setSelectedDiameter(diameter);
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 0.95,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const handleProceed = () => {
    if (selectedDiameter) {
      navigation.navigate('ImageSource', { diameter: selectedDiameter });
    }
  };

  const handleGoBack = () => {
    navigation.goBack();
  };

  const selectedDiameterData = diameters.find(d => d.value === selectedDiameter);
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
        style={styles.content}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Section */}
        <Animated.View
          style={[
            styles.headerContent,
            {
              opacity: fadeAnim,
              transform: [{ scale: pulseAnim }],
            },
          ]}
        >
          {/* Back Button */}
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleGoBack}
          >
            <Icon name="arrow-back" size={24} color="#00D4FF" />
          </TouchableOpacity>

          <View style={styles.logoContainer}>
            <Icon name="straighten" size={50} color="#00D4FF" />
          </View>
          <Text style={styles.headerTitle}>Select TMT Bar Diameter</Text>
          <Text style={styles.headerSubtitle}>
            Choose the diameter for AI-powered analysis
          </Text>
        </Animated.View>

        {/* Diameter Grid */}
        <Animated.View 
          style={[
            styles.diameterGrid,
            {
              opacity: fadeAnim,
            },
          ]}
        >
          {diameters.map((diameter, index) => (
            <Animated.View
              key={diameter.value}
              style={[
                styles.cardContainer,
                {
                  opacity: fadeAnim,
                  transform: [
                    {
                      translateY: fadeAnim.interpolate({
                        inputRange: [0, 1],
                        outputRange: [50, 0],
                      }),
                    },
                  ],
                },
              ]}
            >
              <TouchableOpacity
                style={[
                  styles.diameterCard,
                  selectedDiameter === diameter.value && styles.selectedCard,
                ]}
                onPress={() => handleDiameterSelect(diameter.value)}
                activeOpacity={0.8}
              >
                {/* Glow Effect */}
                {selectedDiameter === diameter.value && (
                  <Animated.View
                    style={[
                      styles.cardGlow,
                      {
                        backgroundColor: diameter.glowColor,
                        opacity: glowOpacity,
                      },
                    ]}
                  />
                )}

                <LinearGradient
                  colors={selectedDiameter === diameter.value ? diameter.color : ['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.05)']}
                  style={styles.cardGradient}
                >
                  <Animated.View
                    style={[
                      styles.cardContent,
                      selectedDiameter === diameter.value && {
                        transform: [{ scale: scaleAnim }],
                      },
                    ]}
                  >
                    <View style={[
                      styles.iconContainer,
                      selectedDiameter === diameter.value && styles.selectedIconContainer,
                    ]}>
                      <Icon 
                        name={diameter.icon} 
                        size={40} 
                        color={selectedDiameter === diameter.value ? 'white' : '#00D4FF'} 
                      />
                    </View>
                    <Text style={[
                      styles.diameterLabel,
                      selectedDiameter === diameter.value && styles.selectedLabel,
                    ]}>
                      {diameter.label}
                    </Text>
                    <View style={styles.rimInfo}>
                      <Text style={[
                        styles.rimLabel,
                        selectedDiameter === diameter.value && styles.selectedRimLabel,
                      ]}>
                        Rim Thickness
                      </Text>
                      <Text style={[
                        styles.rimValue,
                        selectedDiameter === diameter.value && styles.selectedRimValue,
                      ]}>
                        {diameter.rimMin} - {diameter.rimMax} mm
                      </Text>
                    </View>
                    {selectedDiameter === diameter.value && (
                      <View style={styles.selectedIndicator}>
                        <Icon name="check-circle" size={24} color="#4CAF50" />
                      </View>
                    )}
                  </Animated.View>
                </LinearGradient>
              </TouchableOpacity>
            </Animated.View>
          ))}
        </Animated.View>

        {selectedDiameter && (
          <Animated.View 
            style={[
              styles.infoCard,
              {
                opacity: fadeAnim,
                transform: [
                  {
                    translateY: fadeAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [20, 0],
                    }),
                  },
                ],
              },
            ]}
          >
            <Icon name="info" size={24} color="#00D4FF" />
            <View style={styles.infoContent}>
              <Text style={styles.infoTitle}>Selected: {selectedDiameterData.label}</Text>
              <Text style={styles.infoText}>
                Expected rim thickness: {selectedDiameterData.rimMin} - {selectedDiameterData.rimMax} mm
              </Text>
              <Text style={styles.infoText}>
                This represents 7-10% of the bar diameter
              </Text>
            </View>
          </Animated.View>
        )}

        {/* Footer Section */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={[
            styles.proceedButton,
            !selectedDiameter && styles.disabledButton,
          ]}
          onPress={handleProceed}
          disabled={!selectedDiameter}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={selectedDiameter ? ['#00D4FF', '#0099CC', '#006699'] : ['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.05)']}
            style={styles.proceedGradient}
          >
            <Text style={styles.proceedButtonText}>
              {selectedDiameter ? 'PROCEED TO ANALYSIS' : 'SELECT A DIAMETER'}
            </Text>
            {selectedDiameter && <Icon name="arrow-forward" size={20} color="white" />}
          </LinearGradient>
        </TouchableOpacity>
      </View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
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
  content: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
    paddingBottom: 30,
  },
  headerContent: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 30,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
    letterSpacing: 1,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    lineHeight: 22,
  },
  diameterGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  cardContainer: {
    width: (width - 60) / 2,
    marginBottom: 20,
  },
  diameterCard: {
    position: 'relative',
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  selectedCard: {
    elevation: 12,
    shadowOpacity: 0.5,
  },
  cardGlow: {
    position: 'absolute',
    top: -10,
    left: -10,
    right: -10,
    bottom: -10,
    borderRadius: 25,
    zIndex: -1,
  },
  cardGradient: {
    padding: 20,
    alignItems: 'center',
    minHeight: 160,
  },
  cardContent: {
    alignItems: 'center',
    width: '100%',
  },
  iconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
    borderWidth: 2,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  selectedIconContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: 'rgba(255, 255, 255, 0.5)',
  },
  diameterLabel: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 10,
    textAlign: 'center',
  },
  selectedLabel: {
    color: '#FFFFFF',
  },
  rimInfo: {
    alignItems: 'center',
  },
  rimLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    marginBottom: 4,
  },
  selectedRimLabel: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  rimValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#00D4FF',
  },
  selectedRimValue: {
    color: '#FFFFFF',
  },
  selectedIndicator: {
    position: 'absolute',
    top: 10,
    right: 10,
  },
  infoCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    flexDirection: 'row',
    alignItems: 'flex-start',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  infoContent: {
    flex: 1,
    marginLeft: 15,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    lineHeight: 20,
    marginBottom: 4,
  },
  footer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  proceedButton: {
    borderRadius: 30,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  disabledButton: {
    elevation: 4,
    shadowOpacity: 0.1,
  },
  proceedGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 30,
  },
  proceedButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginRight: 10,
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
});

export default DiameterSelectionScreen;
