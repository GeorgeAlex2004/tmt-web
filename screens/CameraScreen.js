import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Dimensions,
  ActivityIndicator,
  Platform,
  Image,
} from 'react-native';
import { CameraView } from 'expo-camera';
import { useCameraPermissions } from 'expo-camera';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';
import { BACKEND_URL } from '../config';
import LoadingAnimation from '../components/LoadingAnimation';

const { width: screenWidth } = Dimensions.get('window');

const CameraScreen = ({ route, navigation }) => {
  const { diameter } = route.params;
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [facing, setFacing] = useState('back');
  const [flash, setFlash] = useState('off');
  const [isCameraReady, setIsCameraReady] = useState(false);
  const cameraRef = useRef(null);

  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [mediaPermission, requestMediaPermission] = MediaLibrary.usePermissions();

  const onCameraReady = () => setIsCameraReady(true);

  const takePicture = async () => {
    if (cameraRef.current && !isCapturing && isCameraReady) {
      setIsCapturing(true);
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.9,
          base64: true,
          exif: true,
        });
        setCapturedImage(photo);
        await processImage(photo);
      } catch (error) {
        console.error('Camera error:', error);
        Alert.alert('Error', 'Failed to capture image. Please try again.');
        setIsCapturing(false);
      }
    }
  };

  // Upload image to backend and navigate to ResultsScreen
  const processImage = async (imageData) => {
    try {
      console.log('Starting image processing...');
      console.log('Backend URL:', BACKEND_URL);
      
      // Compress and resize before upload
      const manipResult = await ImageManipulator.manipulateAsync(
        imageData.uri,
        [{ resize: { width: 512 } }], // Resize to 512px width
        { compress: 0.6, format: ImageManipulator.SaveFormat.JPEG }
      );

      console.log('Image compressed, preparing upload...');

      // Optionally save to gallery
      if (Platform.OS !== 'web' && mediaPermission?.granted) {
        await MediaLibrary.saveToLibraryAsync(manipResult.uri);
      }

      // Prepare FormData for upload
      const formData = new FormData();
      formData.append('image', {
        uri: manipResult.uri,
        type: 'image/jpeg',
        name: 'photo.jpg',
      });
      formData.append('diameter', diameter);

      console.log('Uploading to:', `${BACKEND_URL}/process-ring-test`);

      // Upload to backend with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout for SAM processing

      const response = await fetch(`${BACKEND_URL}/process-ring-test`, {
        method: 'POST',
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

      const result = await response.json();

      if (!result.test_id) {
        throw new Error('No test_id returned from backend.');
      }

      console.log('Processing completed successfully');

      setIsCapturing(false);

      // Navigate to ResultsScreen with testId and imageUri
      navigation.navigate('Results', {
        diameter,
        imageUri: manipResult.uri,
        testId: result.test_id,
        analysisData: result, // Pass the complete analysis data
      });
    } catch (error) {
      console.error('Processing error:', error);
      Alert.alert('Error', error.message || 'Failed to process image. Please try again.');
      setIsCapturing(false);
    }
  };

  const toggleFlash = () => {
    setFlash(current => (current === 'off' ? 'on' : 'off'));
  };

  if (!cameraPermission || !mediaPermission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0D47A1" />
        <Text style={styles.permissionText}>Requesting permissions...</Text>
      </View>
    );
  }

  if (!cameraPermission.granted || !mediaPermission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>Camera and media permissions are required.</Text>
        {!cameraPermission.granted && (
          <TouchableOpacity style={styles.permissionButton} onPress={requestCameraPermission}>
            <Text style={styles.permissionButtonText}>Grant Camera Permission</Text>
          </TouchableOpacity>
        )}
        {!mediaPermission.granted && (
          <TouchableOpacity style={styles.permissionButton} onPress={requestMediaPermission}>
            <Text style={styles.permissionButtonText}>Grant Media Permission</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        flash={flash}
        onCameraReady={onCameraReady}
        ratio="16:9"
      >
        <View style={styles.overlayContainer}>
          <LinearGradient colors={['rgba(0,0,0,0.7)', 'transparent']} style={styles.topBar}>
            <Text style={styles.instructionText}>
              Align TMT bar cross-section with the circle
            </Text>
            <Text style={styles.diameterText}>{diameter}mm Bar Selected</Text>
          </LinearGradient>

          <View style={styles.centerContainer}>
            {(diameter === 12 || diameter === 16 || diameter === 20 || diameter === 25) && (
              <Image
                source={
                  diameter === 12
                    ? require('../assets/overlays/overlay12.png')
                    : diameter === 16
                    ? require('../assets/overlays/overlay16.png')
                    : diameter === 20
                    ? require('../assets/overlays/overlay20.png')
                    : require('../assets/overlays/overlay25.png')
                }
                style={{
                  width: screenWidth * 0.7,
                  height: screenWidth * 0.7,
                  opacity: 0.5,
                  resizeMode: 'contain',
                  position: 'absolute',
                }}
              />
            )}
          </View>
          <View style={styles.bottomControls}>
            <TouchableOpacity style={styles.controlButton} onPress={toggleFlash}>
              <Icon name={flash === 'on' ? 'flash-on' : 'flash-off'} size={24} color="white" />
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.captureButton, !isCameraReady && styles.buttonDisabled]}
              onPress={takePicture}
              disabled={!isCameraReady || isCapturing}
            >
              {isCapturing ? (
                <ActivityIndicator size="large" color="white" />
              ) : (
                <View style={styles.captureButtonInner} />
              )}
            </TouchableOpacity>

            <View style={styles.controlButton} />
          </View>
        </View>
      </CameraView>
      
      {/* Loading Animation */}
      <LoadingAnimation 
        visible={isCapturing}
        title="Processing Image..."
        subtitle="SAM model processing may take 2-3 minutes"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    flex: 1,
  },
  topBar: {
    paddingTop: 40,
    paddingBottom: 20,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  instructionText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  diameterText: {
    color: '#4CAF50',
    fontSize: 14,
    marginTop: 5,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 40,
    paddingHorizontal: 30,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#4CAF50',
  },
  permissionText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 20,
  },
  permissionButton: {
    marginTop: 20,
    backgroundColor: '#0D47A1',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
  },
});

export default CameraScreen;
