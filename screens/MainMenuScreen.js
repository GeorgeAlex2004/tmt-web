// ==================== screens/MainMenuScreen.js ====================
import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Animated,
  Dimensions,
  Image,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from '../components/Icon';

const { width } = Dimensions.get('window');

const MainMenuScreen = ({ navigation }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
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
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 50,
        friction: 8,
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

  const menuItems = [
    {
      id: 1,
      title: 'Ring Test',
      subtitle: 'AI-Powered TMT Analysis',
      icon: 'science',
      color: ['#00D4FF', '#0099CC', '#006699'],
      glowColor: 'rgba(0, 212, 255, 0.3)',
      onPress: () => navigation.navigate('DiameterSelection'),
    },
    {
      id: 2,
      title: 'Test History',
      subtitle: 'Analytics & Reports',
      icon: 'analytics',
      color: ['#FF6B6B', '#FF5252', '#D32F2F'],
      glowColor: 'rgba(255, 107, 107, 0.3)',
      onPress: () => navigation.navigate('History'),
    },
    {
      id: 3,
      title: 'Rib Test',
      subtitle: 'Advanced Rib Analysis',
      icon: 'straighten',
      color: ['#4ECDC4', '#26A69A', '#00897B'],
      glowColor: 'rgba(78, 205, 196, 0.3)',
      onPress: () => navigation.navigate('RibTest'),
    },
    {
      id: 4,
      title: 'Settings',
      subtitle: 'System Configuration',
      icon: 'tune',
      color: ['#A78BFA', '#8B5CF6', '#7C3AED'],
      glowColor: 'rgba(167, 139, 250, 0.3)',
      onPress: () => navigation.navigate('Settings'),
    },
  ];

  const glowOpacity = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 1],
  });

  const renderMenuItem = (item, index) => {
    const animatedStyle = {
      opacity: fadeAnim,
      transform: [
        { scale: scaleAnim },
        {
          translateY: fadeAnim.interpolate({
            inputRange: [0, 1],
            outputRange: [50, 0],
          }),
        },
      ],
    };

    return (
      <Animated.View
        key={item.id}
        style={[styles.menuItemContainer, animatedStyle]}
      >
        <TouchableOpacity
          style={styles.menuItem}
          onPress={item.onPress}
          activeOpacity={0.8}
        >
          <Animated.View
            style={[
              styles.menuGlow,
              {
                backgroundColor: item.glowColor,
                opacity: glowOpacity,
              },
            ]}
          />
          <LinearGradient
            colors={item.color}
            style={styles.menuGradient}
            start={{x: 0, y: 0}}
            end={{x: 1, y: 1}}
          >
            <View style={styles.iconContainer}>
              <Icon name={item.icon} size={35} color="white" />
            </View>
            <Text style={styles.menuTitle}>{item.title}</Text>
            <Text style={styles.menuSubtitle}>{item.subtitle}</Text>
            <View style={styles.menuArrow}>
              <Icon name="arrow-forward" size={20} color="rgba(255, 255, 255, 0.8)" />
            </View>
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  return (
      <LinearGradient
        colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
      >
        {/* Animated Background Elements */}
        <Animated.View style={[styles.backgroundCircle, { opacity: glowOpacity }]} />
        <Animated.View style={[styles.backgroundCircle2, { opacity: glowOpacity }]} />
        
      <ScrollView 
        style={styles.menuContainer}
        contentContainerStyle={styles.menuContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Section */}
        <View style={styles.headerContent}>
          <Animated.View
            style={[
              styles.logoContainer,
              {
                transform: [{ scale: pulseAnim }],
              },
            ]}
          >
            <Icon name="precision-manufacturing" size={50} color="#00D4FF" />
          </Animated.View>
          <Text style={styles.welcomeText}>Welcome to</Text>
          <Text style={styles.headerTitle}>TMT Quality Testing</Text>
          <Text style={styles.headerSubtitle}>Advanced AI-Powered Analysis System</Text>
        </View>

        {/* Menu Grid */}
        <View style={styles.gridContainer}>
          {menuItems.map((item, index) => renderMenuItem(item, index))}
        </View>
        
        {/* Partner Logos Section */}
        <View style={styles.logosContainer}>
          <View style={styles.logosRow}>
            <Image 
              source={require('../assets/images/tata_steel.png')} 
              style={styles.logoImage}
              resizeMode="contain"
              onError={(e) => {
                console.log('Tata Steel logo error:', e);
                // Fallback for web if require doesn't work
                if (Platform.OS === 'web') {
                  return { uri: 'https://via.placeholder.com/200x80/0066CC/FFFFFF?text=TATA+STEEL' };
                }
              }}
            />
            <Image 
              source={require('../assets/images/SRM_logo.png')} 
              style={[styles.logoImage, styles.srmLogoSmall]}
              resizeMode="contain"
              onError={(e) => {
                console.log('SRM logo error:', e);
                if (Platform.OS === 'web') {
                  return { uri: 'https://via.placeholder.com/150x60/CC0000/FFFFFF?text=SRM' };
                }
              }}
            />
            <Image 
              source={require('../assets/images/Coe.png')} 
              style={styles.logoImage}
              resizeMode="contain"
              onError={(e) => {
                console.log('Coe logo error:', e);
                if (Platform.OS === 'web') {
                  return { uri: 'https://via.placeholder.com/200x80/009900/FFFFFF?text=CoE' };
                }
              }}
            />
          </View>
        </View>

        {/* Logout Button */}
        <View style={{ alignItems: 'center', marginTop: 20, marginBottom: 30 }}>
          <TouchableOpacity
            style={styles.logoutButton}
            onPress={() => navigation.replace('Login')}
            activeOpacity={0.85}
          >
            <LinearGradient
              colors={['#FF6B6B', '#FF5252', '#D32F2F']}
              style={styles.logoutGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Icon name="logout" size={22} color="white" style={{ marginRight: 8 }} />
              <Text style={styles.logoutButtonText}>LOG OUT</Text>
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
  menuContainer: {
    flex: 1,
  },
  menuContent: {
    padding: 20,
    paddingTop: 60,
  },
  headerContent: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 30,
  },
  logoContainer: {
    width: 120,
    height: 120,
    borderRadius: 40,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  welcomeText: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: '500',
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: '900',
    color: '#FFFFFF',
    marginTop: 8,
    letterSpacing: 2,
    textShadowColor: 'rgba(0, 212, 255, 0.5)',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
    textAlign: 'center',
    maxWidth: '90%',
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#00D4FF',
    marginTop: 8,
    fontWeight: '600',
    letterSpacing: 1,
    textAlign: 'center',
    maxWidth: '90%',
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  menuItemContainer: {
    width: (width - 60) / 2,
    marginBottom: 20,
  },
  menuItem: {
    position: 'relative',
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 10,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  menuGlow: {
    position: 'absolute',
    top: -10,
    left: -10,
    right: -10,
    bottom: -10,
    borderRadius: 25,
    zIndex: -1,
  },
  menuGradient: {
    padding: 25,
    alignItems: 'center',
    minHeight: 140,
    position: 'relative',
  },
  iconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  menuTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
    textAlign: 'center',
  },
  menuSubtitle: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    lineHeight: 16,
  },
  menuArrow: {
    position: 'absolute',
    top: 15,
    right: 15,
  },
  statsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 20,
    padding: 25,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  statsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    justifyContent: 'center',
  },
  statsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginLeft: 10,
    letterSpacing: 1,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
    position: 'relative',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: '900',
    color: '#00D4FF',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  statGlow: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    top: -5,
    left: -5,
    zIndex: -1,
  },
  logosContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 20,
    padding: 25,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
    marginTop: 20,
  },
  logosTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 20,
  },
  logosRow: {
    flexDirection: 'row',
    flexWrap: 'wrap', // allow wrapping on small screens
    justifyContent: 'center', // center logos
    alignItems: 'center', // vertical alignment
    gap: 20, // spacing between logos (if supported)
    marginTop: 10,
    marginBottom: 10,
  },
  logoImage: {
    width: undefined,
    height: 180, // increased from 60 to 180 for exponential size increase
    aspectRatio: 2.5, // keep logo aspect ratio
    marginHorizontal: 25, // more spacing for larger logos
    marginVertical: 20,
    maxWidth: 360, // increased max width
    resizeMode: 'contain',
  },
  srmLogoSmall: {
    height: 120,
    maxWidth: 180,
  },
  logoutButton: {
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#FF6B6B',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    width: 220,
  },
  logoutGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 30,
  },
  logoutButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    letterSpacing: 1,
  },
});

export default MainMenuScreen;
