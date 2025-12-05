import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
  Alert,
  Animated,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Platform } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { BACKEND_URL } from '../config';
import * as FileSystem from 'expo-file-system/legacy';
import { ApiService } from '../src/services/ApiService';
import LoadingAnimation from '../components/LoadingAnimation';

const { width } = Dimensions.get('window');

const ImageSourceScreen = ({ route, navigation }) => {
  const { diameter } = route.params;
  const [isProcessing, setIsProcessing] = useState(false);
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

  const handleCameraPress = () => {
    navigation.navigate('Camera', { diameter });
  };

  const testImagePicker = async () => {
    try {
      console.log('Testing ImagePicker...');
      console.log('ImagePicker object:', ImagePicker);
      console.log('Available methods:', Object.keys(ImagePicker));
      
      // Check what properties are available
      const properties = [
        'MediaType',
        'MediaTypeOptions', 
        'launchImageLibraryAsync',
        'requestMediaLibraryPermissionsAsync'
      ];
      
      properties.forEach(prop => {
        console.log(`${prop}:`, ImagePicker[prop]);
      });
      
      // Test if we can request permissions
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      console.log('Permission status:', status);
      
      Alert.alert('Success', 'ImagePicker is working correctly! Check console for available APIs.');
    } catch (error) {
      console.error('ImagePicker test error:', error);
      Alert.alert('Error', `ImagePicker test failed: ${error.message}`);
    }
  };

  const handleGalleryPress = async () => {
    try {
      console.log('Opening gallery picker...');
      
      // Request media library permissions
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      console.log('Media library permission status:', status);
      
      if (status !== 'granted') {
        alert('Sorry, we need camera roll permissions to make this work!');
        return;
      }

      // Launch image picker without specifying mediaTypes (let it default to images)
      const result = await ImagePicker.launchImageLibraryAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
      });

      console.log('Image picker result:', result);

      if (!result.canceled && result.assets && result.assets.length > 0) {
        // Process the selected image
        await processGalleryImage(result.assets[0]);
      } else {
        console.log('No image selected or picker was canceled');
      }
    } catch (error) {
      console.error('Gallery error:', error);
      alert(`Failed to select image from gallery: ${error.message}`);
    }
  };

  const testSimpleGallery = async () => {
    try {
      console.log('Testing simple gallery picker...');
      
      // Request permissions
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant media library permissions');
        return;
      }

      // Test without specifying mediaTypes
      const result = await ImagePicker.launchImageLibraryAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
      });

      console.log('Simple picker result:', result);
      
      if (!result.canceled && result.assets && result.assets.length > 0) {
        Alert.alert('Success', 'Image selected successfully!');
        // Process the image
        await processGalleryImage(result.assets[0]);
      } else {
        Alert.alert('Info', 'No image selected');
      }
    } catch (error) {
      console.error('Simple gallery test error:', error);
      Alert.alert('Error', `Simple gallery test failed: ${error.message}`);
    }
  };

  const processGalleryImage = async (imageAsset) => {
    try {
      setIsProcessing(true);
      
      console.log('Starting image processing...');
      console.log('Backend URL:', BACKEND_URL);
      
      // Compress and resize the image
      const manipResult = await ImageManipulator.manipulateAsync(
        imageAsset.uri,
        [{ resize: { width: 512 } }],
        { compress: 0.6, format: ImageManipulator.SaveFormat.JPEG }
      );

      // Web cannot use FileSystem.readAsStringAsync; use FormData upload on web
      let result;
      if (Platform.OS === 'web') {
        console.log('Web platform detected. Uploading via FormData...');
        const formData = new FormData();
        formData.append('image', {
          uri: manipResult.uri,
          type: 'image/jpeg',
          name: 'photo.jpg',
        });
        formData.append('diameter', diameter.toString());

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000);
        const response = await fetch(`${BACKEND_URL}/process-ring-test`, {
          method: 'POST',
          headers: { Accept: 'application/json' },
          body: formData,
          signal: controller.signal,
        }).catch((error) => {
          if (error.name === 'AbortError') {
            throw new Error('Request timed out. Please check your network connection.');
          }
          throw new Error(`Network error: ${error.message}. Please ensure:\n1. Backend is running\n2. IP address is correct\n3. Phone and laptop are on same WiFi`);
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Backend error (${response.status}): ${errorText}`);
        }
        result = await response.json();
      } else {
        console.log('Reading image as base64 for ApiService...');
        const imageBase64 = await FileSystem.readAsStringAsync(manipResult.uri, { encoding: 'base64' });
        console.log('Calling ApiService.processRingTest...');
        result = await ApiService.processRingTest({ imageBase64, diameter });
      }

      if (!result?.test_id) {
        throw new Error('No test_id returned from backend.');
      }

      setIsProcessing(false);

      // Navigate to ResultsScreen
      navigation.navigate('Results', {
        diameter,
        imageUri: manipResult.uri,
        testId: result.test_id,
        analysisData: result,
      });
    } catch (error) {
      console.error('Processing error:', error);
      alert(error.message || 'Failed to process image. Please try again.');
      setIsProcessing(false);
    }
  };

  const handleGoBack = () => {
    navigation.goBack();
  };

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
            <Icon name="camera-alt" size={50} color="#00D4FF" />
          </View>
          <Text style={styles.headerTitle}>Choose Image Source</Text>
          <Text style={styles.headerSubtitle}>
            Selected diameter: {diameter}mm
          </Text>
        </Animated.View>

        {/* Option Cards */}
        <Animated.View
          style={[
            styles.optionCard,
            {
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
            style={styles.cardButton}
            onPress={handleCameraPress}
            activeOpacity={0.8}
          >
            <Animated.View
              style={[
                styles.cardGlow,
                {
                  backgroundColor: 'rgba(0, 212, 255, 0.3)',
                  opacity: glowOpacity,
                },
              ]}
            />
            <LinearGradient
              colors={['#00D4FF', '#0099CC', '#006699']}
              style={styles.cardGradient}
            >
              <View style={styles.iconContainer}>
                <Icon name="camera-alt" size={60} color="white" />
              </View>
              <Text style={styles.optionTitle}>Take Photo</Text>
              <Text style={styles.optionDescription}>
                Use camera to capture TMT bar cross-section
              </Text>
              <View style={styles.cardArrow}>
                <Icon name="arrow-forward" size={20} color="rgba(255, 255, 255, 0.8)" />
              </View>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>

        <Animated.View
          style={[
            styles.optionCard,
            {
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
            style={styles.cardButton}
            onPress={handleGalleryPress}
            activeOpacity={0.8}
          >
            <Animated.View
              style={[
                styles.cardGlow,
                {
                  backgroundColor: 'rgba(255, 107, 107, 0.3)',
                  opacity: glowOpacity,
                },
              ]}
            />
            <LinearGradient
              colors={['#FF6B6B', '#FF5252', '#D32F2F']}
              style={styles.cardGradient}
            >
              <View style={styles.iconContainer}>
                <Icon name="photo-library" size={60} color="white" />
              </View>
              <Text style={styles.optionTitle}>Choose from Gallery</Text>
              <Text style={styles.optionDescription}>
                Select an existing image from your device
              </Text>
              <View style={styles.cardArrow}>
                <Icon name="arrow-forward" size={20} color="rgba(255, 255, 255, 0.8)" />
              </View>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>

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
            <Text style={styles.infoText}>
              For best results, ensure the TMT bar cross-section is clearly visible and well-lit
            </Text>
          </View>
        </Animated.View>
      </ScrollView>

      {/* Loading Animation */}
      <LoadingAnimation 
        visible={isProcessing}
        title="Processing Image..."
        subtitle="SAM model processing may take 2-3 minutes"
      />
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
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
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
    color: '#00D4FF',
    textAlign: 'center',
    marginTop: 8,
    fontWeight: '600',
    letterSpacing: 1,
  },
  optionCard: {
    marginBottom: 25,
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 10,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  cardButton: {
    position: 'relative',
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
    padding: 30,
    alignItems: 'center',
    position: 'relative',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
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
  optionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 10,
    textAlign: 'center',
    letterSpacing: 1,
  },
  optionDescription: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    lineHeight: 22,
  },
  cardArrow: {
    position: 'absolute',
    top: 20,
    right: 20,
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    borderRadius: 15,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
    alignItems: 'center',
  },
  infoContent: {
    flex: 1,
    marginLeft: 15,
  },
  infoText: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    lineHeight: 22,
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

export default ImageSourceScreen; 