import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Image,
  Animated,
  Dimensions,
  Modal,
} from 'react-native';
import { PanGestureHandler, State, GestureHandlerRootView } from 'react-native-gesture-handler';
import Slider from '@react-native-community/slider';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { AnalysisStep } from '../models/AnalysisStep';
import { ApiService } from '../services/ApiService';
import { LinearGradient } from 'expo-linear-gradient';
import { Camera, CameraView as ExpoCameraView } from 'expo-camera';


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

interface TmtCameraViewProps {
  onAnalysisComplete: (results: any) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
  diameter: number;
  rows: number;
}



const CameraView: React.FC<TmtCameraViewProps> = ({
  onAnalysisComplete,
  isAnalyzing,
  setIsAnalyzing,
  diameter,
  rows,
}) => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isServerAvailable, setIsServerAvailable] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any | null>(null);
  const [tmtDetectionResult, setTmtDetectionResult] = useState<any | null>(null);
  const [showSegmentedImage, setShowSegmentedImage] = useState(false);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [lastImageData, setLastImageData] = useState<string | null>(null); // Store last image data for analysis
  const [detectionProgress, setDetectionProgress] = useState<string>(''); // Show detection progress
  const [isYoloDetecting, setIsYoloDetecting] = useState(false); // Show YOLO detection overlay
  const [quickDetectionResults, setQuickDetectionResults] = useState<{
    ribCount: number;
    tmtBars: number;
    totalRibs: number;
    inferenceTime: number;
    tmtBoundingBox: [number, number, number, number] | null;
    analysisType?: 'angle_length' | 'height' | 'unified'; // Type of analysis to perform
  } | null>(null); // Store quick detection results
  const [showCalibrationUI, setShowCalibrationUI] = useState(false); // Show calibration UI
  const { width: screenWidth, height: screenHeight } = Dimensions.get('window');



  // Live camera state
  const cameraRef = useRef<ExpoCameraView | null>(null);
  const [cameraPermission, setCameraPermission] = useState<any>(null);
  const [cameraType, setCameraType] = useState<'front' | 'back'>('back');
  const [showLiveCamera, setShowLiveCamera] = useState(false);

  // Guide overlay state - using static require for bundled asset
  const overlaySource = require('../../assets/images/latest.jpg');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.4);
  const [isOverlayVisible, setIsOverlayVisible] = useState<boolean>(true);

  // Animation values
  const loadingAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    checkServerStatus();
    const serverCheckTimer = setInterval(() => {
      checkServerStatus();
    }, 10000);

    return () => {
      clearInterval(serverCheckTimer);
    };
  }, []);

  // Reset camera view when step changes
  useEffect(() => {
    setCapturedImage(null);
    setImageUri(null);
    setTmtDetectionResult(null);
    setShowSegmentedImage(false);
    setIsYoloDetecting(false);
  }, []);

  useEffect(() => {
    if (isAnalyzing) {
      animateLoading();
    }
  }, [isAnalyzing]);



  const checkServerStatus = async () => {
    try {
      const isAvailable = await ApiService.isServerAvailable();
      setIsServerAvailable(isAvailable);
    } catch (error) {
      setIsServerAvailable(false);
    }
  };

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Camera permission is required to take photos');
      return false;
    }
    return true;
  };

  const openLiveCamera = async () => {
    try {
      const { status } = await Camera.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera permission is required to take photos.');
        return;
      }
      setCameraPermission({ granted: true });
      setShowLiveCamera(true);
    } catch (error) {
      console.error('Error opening camera:', error);
      Alert.alert('Error', 'Failed to open camera. Please try again.');
    }
  };

  const captureFromLiveCamera = async () => {
    try {
      if (cameraRef.current) {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: true,
        });
        if (photo?.uri) {
          setImageUri(photo.uri);
          setCapturedImage(photo.uri);
          if (photo.base64) {
            setLastImageData(photo.base64);
          }
          setShowLiveCamera(false);
        }
      }
    } catch (error) {
      console.error('Error capturing photo:', error);
      Alert.alert('Error', 'Failed to capture photo. Please try again.');
    }
  };

  const takePhoto = async () => {
    // Use live camera first, fallback to system camera
    await openLiveCamera();
  };

  const pickImage = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Gallery permission is required to select images.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        aspect: [4, 3],
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        setImageUri(asset.uri);
        setCapturedImage(asset.uri);
        if (asset.base64) {
          setLastImageData(asset.base64);
        }
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
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

  const startCalibration = async () => {
    if (!quickDetectionResults || !lastImageData || !diameter) {
      setDetectionError('Quick detection results not available. Please try again.');
      return;
    }
    
    console.log('=== STEP 3: Starting SAM Calibration ===');
    setIsAnalyzing(true);
    setShowCalibrationUI(false);
    setDetectionProgress('🔍 Starting SAM segmentation for calibration...');
    
    try {
      const tmtResult = await ApiService.detectTmtBar({
        imageBase64: lastImageData,
        diameter: diameter,
        tmt_bounding_box: quickDetectionResults.tmtBoundingBox  // Pass the TMT bounding box
      });
      console.log('SAM calibration result:', tmtResult);
      
      if (!tmtResult) {
        setDetectionError('SAM calibration failed: No response from server. Please try again.');
        setIsYoloDetecting(false);
        setIsAnalyzing(false);
        return;
      }
      
      if (tmtResult.status === 'error') {
        const errorMessage = tmtResult.error || 'Unknown error';
        setDetectionError(`SAM calibration failed: ${errorMessage}. Please try again.`);
        setIsYoloDetecting(false);
        setIsAnalyzing(false);
        return;
      }

      // SAM calibration successful, show segmented image
      setTmtDetectionResult(tmtResult);
      setShowSegmentedImage(true);
      setDetectionProgress('✅ SAM calibration complete! Segmented TMT bar displayed.');
      
      // Stop analyzing here - show segmented image and wait for user to press analyze
      setIsAnalyzing(false);
      setShowSegmentedImage(true);
      setDetectionProgress('✅ SAM calibration complete! Now press "Analyze" to proceed with analysis.');
      
    } catch (error: any) {
      console.error('SAM calibration failed:', error);
      setDetectionError(`SAM calibration failed: ${error.message || 'Unknown error'}. Please try again.`);
      setIsYoloDetecting(false);
      setIsAnalyzing(false);
    }
  };

  const startAnalysisAfterCalibration = async () => {
    // STRICT VALIDATION: Must have successful SAM calibration
    if (!tmtDetectionResult || !diameter) {
      setDetectionError('[ERROR] SAM calibration required! Please complete SAM calibration first.');
      return;
    }
    
    // Validate that SAM calibration was successful
    if (!tmtDetectionResult.used_scale_factor || !tmtDetectionResult.scale_factor_method) {
      setDetectionError('[ERROR] SAM calibration incomplete! Scale factor not calculated. Please restart SAM calibration.');
      return;
    }
    
    // Validate that we have a segmented image
    if (!tmtDetectionResult.cropped_image) {
      setDetectionError('[ERROR] SAM segmentation failed! No segmented image available. Please restart SAM calibration.');
      return;
    }
    
    console.log('=== STEP 4: Starting Analysis After SAM Calibration ===');
    console.log('✅ SAM calibration validated successfully');
    console.log('✅ Scale factor:', tmtDetectionResult.used_scale_factor);
    console.log('✅ Calibration method:', tmtDetectionResult.scale_factor_method);
    console.log('✅ Segmented image available');
    
    setIsAnalyzing(true);
    setDetectionProgress('🔍 Starting unified rib analysis (angle, height, length, interdistance)...');
    
    try {
      // Use the segmented image and calibration data from SAM
      console.log('Using SAM calibration data for analysis...');
      console.log('Scale factor from calibration:', tmtDetectionResult.used_scale_factor);
      console.log('Scale factor method:', tmtDetectionResult.scale_factor_method);
      
      // Use the unified rib analysis API for complete analysis
      console.log('Calling analyzeRibUnified API with pre-segmented image for complete rib analysis...');
      console.log('DEBUG: segmentedImageBase64 length:', tmtDetectionResult.cropped_image ? tmtDetectionResult.cropped_image.length : 'null');
      console.log('DEBUG: tmtDetectionResult keys:', Object.keys(tmtDetectionResult));
      
      const results = await ApiService.analyzeRibUnified({
        imageBase64: lastImageData,  // Original image (for reference)
        segmentedImageBase64: tmtDetectionResult.cropped_image,  // Pre-segmented image from SAM
        diameter: diameter,
        scale_factor: tmtDetectionResult.used_scale_factor,  // Pass the calibrated scale factor
        calibration_method: tmtDetectionResult.scale_factor_method  // Pass the calibration method
      });
      
      console.log('Unified analysis results received:', results);
      onAnalysisComplete(results);
      setDetectionProgress(''); // Clear progress
      
    } catch (error: any) {
      console.error('Calibrated analysis failed:', error);
      setDetectionError(`[ERROR] Calibrated analysis failed: ${error.message || 'Unknown error'}. Please try again.`);
      setIsYoloDetecting(false);
      setIsAnalyzing(false);
    }
  };

  const analyzeImage = async (base64Data: string) => {
    if (!isServerAvailable) {
      setDetectionError('Server is not available. Please check your connection and try again.');
      setIsAnalyzing(false);
      return;
    }

    let imageBase64 = base64Data;
    if (!imageBase64) {
      if (lastImageData) {
        console.log('Using stored lastImageData for analysis');
        imageBase64 = lastImageData;
      } else if (imageUri) {
        try {
          console.log('Converting imageUri to base64 for analysis');
          imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
            encoding: 'base64',
          });
          console.log('Image converted to base64, length:', imageBase64.length);
        } catch (error) {
          console.error('Error converting image to base64:', error);
          setDetectionError('Failed to process image. Please try taking a new photo.');
          setIsAnalyzing(false);
          return;
        }
      } else {
        console.error('No image data available for analysis');
        setDetectionError('No image data available. Please take a photo first.');
        setIsAnalyzing(false);
        return;
      }
    }

    if (!imageBase64 || imageBase64.length < 1000) {
      setDetectionError('Invalid image data. Please take a new photo.');
      setIsAnalyzing(false);
      return;
    }

    // Log image data for debugging
    console.log('Image data length:', imageBase64.length);
    console.log('Image data preview:', imageBase64.substring(0, 100) + '...');
    
    // Validate and clean image data format
    if (imageBase64.startsWith('data:image/')) {
      // Remove data URL prefix to get pure base64
      imageBase64 = imageBase64.split(',')[1];
      console.log('Removed data URL prefix, new length:', imageBase64.length);
    }
    
    if (!imageBase64 || !imageBase64.match(/^[A-Za-z0-9+/]*={0,2}$/)) {
      console.error('Invalid base64 image format after cleaning');
      setDetectionError('Invalid image format. Please take a new photo.');
      setIsAnalyzing(false);
      return;
    }

    try {
      // UNIFIED: Quick YOLO detection to show rib count immediately
      console.log('=== UNIFIED: Quick YOLO Detection ===');
        setIsYoloDetecting(true); // Show YOLO detection overlay
        setDetectionProgress('🔍 Testing endpoints...');
        
        // Test both endpoints first
        const getEndpointWorking = await ApiService.testQuickEndpoint();
        if (!getEndpointWorking) {
          setDetectionError('Quick detection GET endpoint not available. Please try again.');
          setDetectionProgress('');
          setIsYoloDetecting(false);
          setIsAnalyzing(false);
          return;
        }
        
        setDetectionProgress('🔍 Testing POST with image data...');
        const postEndpointWorking = await ApiService.testSimplePost();
        if (!postEndpointWorking) {
          setDetectionError('Quick detection POST endpoint not working with image data. Please try again.');
          setDetectionProgress('');
          setIsYoloDetecting(false);
          setIsAnalyzing(false);
          return;
        }
        
        setDetectionProgress('🔍 Quick YOLO detection...');
        
        const quickResult = await ApiService.quickDetect({
          imageBase64: imageBase64,
          diameter: diameter
        });
        console.log('Quick detection result:', quickResult);
        
        if (!quickResult || quickResult.status === 'error') {
          const errorMessage = quickResult?.error || 'Quick detection failed';
          setDetectionError(`Quick detection failed: ${errorMessage}. Please try again.`);
          setDetectionProgress(''); // Clear progress
          setIsYoloDetecting(false);
          setIsAnalyzing(false);
          return;
        }
        
        // Show rib count immediately
        const ribCount = quickResult.ribs_in_tmt_regions || 0;
        const tmtBars = quickResult.tmt_bars_detected || 0;
        const totalRibs = quickResult.total_ribs_detected || 0;
        
        console.log(`Quick detection: ${tmtBars} TMT bars, ${totalRibs} total ribs, ${ribCount} ribs in TMT regions`);
        
        // Check if ribs are sufficient
        if (ribCount < 10) {
          setDetectionError(`For proper and accurate analysis results, at least 10 ribs of the TMT bar should be clearly visible. Only ${ribCount} ribs detected. Make sure any damaged ribs are avoided. Please retake the photo.`);
          setDetectionProgress(''); // Clear progress
          setIsYoloDetecting(false); // Hide YOLO overlay
          setIsAnalyzing(false);
          return;
        }
        
        // UNIFIED: Show rib count and calibration button
        console.log('=== UNIFIED: Rib Count Display and Calibration ===');
        setDetectionProgress(`✅ ${ribCount} ribs detected! Ready for calibration.`);
        
        // Store quick detection results and show calibration UI
        setQuickDetectionResults({
          ribCount,
          tmtBars,
          totalRibs,
          inferenceTime: quickResult.inference_time || 0,
          tmtBoundingBox: quickResult.tmt_bounding_box || null,
          analysisType: 'unified' // Mark this as unified analysis
        });
        
        // Stop analyzing here - wait for user to press calibrate
        setIsAnalyzing(false);
        setIsYoloDetecting(false);
        setShowCalibrationUI(true);
        return;
      
      setIsAnalyzing(false);
      setIsYoloDetecting(false);
      
    } catch (error: any) {
      console.error('Analysis failed:', error);
      let errorMessage = 'Analysis failed. Please try again.';
      let errorType = 'processing_error';
      let ribCount = 0;
      
      // Handle different types of errors
      if (error.message) {
        errorMessage = error.message;
        
        // Check for specific error types in the message
        if (error.message.includes('TMT bar not detected') || error.message.includes('tmt_not_found')) {
          errorType = 'tmt_not_found';
        } else if (error.message.includes('insufficient_ribs') || error.message.includes('ribs detected')) {
          errorType = 'insufficient_ribs';
          // Try to extract rib count from error message
          const ribMatch = error.message.match(/(\d+)\s*ribs?/i);
          if (ribMatch) {
            ribCount = parseInt(ribMatch[1]);
          }
        } else if (error.message.includes('Server is not available')) {
          errorType = 'server_unavailable';
        } else if (error.message.includes('Request timed out')) {
          errorType = 'timeout';
        } else if (error.message.includes('NetworkError') || error.message.includes('Could not connect')) {
          errorType = 'connection_error';
        }
      }
      
      // Set appropriate error message based on error type
      if (errorType === 'tmt_not_found') {
        setDetectionError('TMT bar not detected in image, please retake the photo');
      } else if (errorType === 'insufficient_ribs') {
        setDetectionError(`For proper and accurate analysis results, at least 10 ribs of the TMT bar should be clearly visible. Only ${ribCount} ribs detected. Make sure any damaged ribs are avoided. Please retake the photo.`);
      } else if (errorType === 'server_unavailable') {
        setDetectionError('Server is not available. Please ensure the backend server is running and try again.');
      } else if (errorType === 'timeout') {
        setDetectionError('Request timed out. The server is taking too long to respond. Please try again.');
      } else if (errorType === 'connection_error') {
        setDetectionError('Could not connect to the server. Please check your network connection and try again.');
      } else {
        setDetectionError(`Analysis failed: ${errorMessage}. Please try again.`);
      }
      
      setIsAnalyzing(false);
      setIsYoloDetecting(false); // Hide YOLO overlay when error occurs
    }
  };

  const clearResults = () => {
    setCapturedImage(null);
    setImageUri(null);
    setTmtDetectionResult(null);
    setShowSegmentedImage(false);
    setDetectionError(null);
    setQuickDetectionResults(null);
    setShowCalibrationUI(false);
    setDetectionProgress('');
    setIsYoloDetecting(false);
    // Keep overlay selection for the user's convenience
    // Call onAnalysisComplete with clear_results flag to properly reset the parent state
    onAnalysisComplete({ clear_results: true });
  };

  const renderServerStatus = () => {
    return (
      <View style={styles.serverStatusContainer}>
        <View style={[
          styles.serverStatusIndicator,
          { backgroundColor: isServerAvailable ? AppColors.successGreen : AppColors.errorRed }
        ]}>
          <Text style={styles.serverStatusIcon}>
            {isServerAvailable ? '☁️✅' : '☁️❌'}
          </Text>
          <Text style={styles.serverStatusText}>
            {isServerAvailable ? 'Server Connected' : 'Server Offline'}
          </Text>
        </View>
      </View>
    );
  };



  const renderCameraView = () => {
    // Show segmented TMT bar image if available
    if (showSegmentedImage && tmtDetectionResult?.cropped_image) {
      return (
        <View style={styles.segmentedImageContainer}>
          <View style={styles.segmentedImageHeader}>
            <Text style={styles.segmentedImageTitle}>
              [SUCCESS] TMT Bar Detected for Unified Analysis
            </Text>
            <Text style={styles.segmentedImageSubtitle}>
              Segmented TMT bar for complete rib analysis
            </Text>
          </View>
          <Image 
            source={{ uri: `data:image/jpeg;base64,${tmtDetectionResult.cropped_image}` }} 
            style={styles.segmentedImage} 
          />
          <View style={styles.segmentedImageInfo}>
            <Text style={styles.segmentedImageInfoText}>
              Confidence: {(tmtDetectionResult.confidence * 100).toFixed(1)}%
            </Text>
            <Text style={styles.segmentedImageInfoText}>
              Diameter: {tmtDetectionResult.diameter}mm
            </Text>
            <Text style={styles.segmentedImageInfoText}>
              Scale Factor: {tmtDetectionResult.used_scale_factor} pixels/mm
            </Text>
            <Text style={styles.segmentedImageInfoText}>
              Calibration Method: {tmtDetectionResult.scale_factor_method || 'SAM Segmentation'}
            </Text>
            <Text style={[styles.segmentedImageInfoText, { color: '#4CAF50', fontWeight: 'bold' }]}>
              [SUCCESS] SAM Calibration Required for Analysis
            </Text>
          </View>
          <View style={styles.segmentedImageInstructions}>
            <Text style={styles.segmentedImageInstructionsText}>
              [SUCCESS] SAM calibration complete!
            </Text>
            <Text style={styles.segmentedImageInstructionsText}>
              Scale factor: {tmtDetectionResult.used_scale_factor} pixels/mm
            </Text>
            <Text style={styles.segmentedImageInstructionsText}>
              Method: {tmtDetectionResult.scale_factor_method || 'SAM Segmentation'}
            </Text>
            <Text style={styles.segmentedImageInstructionsText}>
              Now press "Complete Rib Analysis" below to proceed with unified analysis.
            </Text>
          </View>
        </View>
      );
    }

    if (!capturedImage) {
      return (
        <View style={styles.cameraPlaceholder}>
          <Text style={styles.cameraPlaceholderText}>📷</Text>
          <Text style={styles.cameraPlaceholderTitle}>Camera Preview</Text>
          <Text style={styles.cameraPlaceholderSubtitle}>
            Complete Rib Analysis
          </Text>
          <Text style={styles.cameraPlaceholderInfo}>
            Take a photo showing the TMT bar for unified analysis (angle, height, length, interdistance)
          </Text>
          <Text style={styles.cameraPlaceholderHint}>
            💡 Capture a clear view of the TMT bar with visible ribs
          </Text>
        </View>
      );
    }

    // After capture: show captured image
    return (
      <View style={styles.imagePreviewContainer}>
        <Image source={{ uri: capturedImage }} style={styles.capturedImage} />
        {detectionError && (
          <View style={styles.detectionErrorOverlay}>
            <View style={styles.detectionErrorContent}>
              <Text style={styles.detectionErrorIcon}>
                {detectionError.includes('TMT bar not detected') ? '🔍' : 
                 detectionError.includes('processing failed') || detectionError.includes('Image processing failed') ? '⚠️' :
                 detectionError.includes('ribs detected') ? '📏' : '❌'}
              </Text>
              <Text style={styles.detectionErrorTitle}>
                {detectionError.includes('TMT bar not detected') ? 'TMT Bar Not Detected' : 
                 detectionError.includes('processing failed') || detectionError.includes('Image processing failed') ? 'Processing Error' :
                 detectionError.includes('ribs detected') ? 'Insufficient Ribs Detected' : 'Analysis Error'}
              </Text>
              <Text style={styles.detectionErrorText}>{detectionError}</Text>
              {detectionError.includes('processing failed') || detectionError.includes('Image processing failed') && (
                <View style={styles.errorSuggestions}>
                  <Text style={styles.errorSuggestionsTitle}>💡 Suggestions:</Text>
                  <Text style={styles.errorSuggestionsText}>• Ensure the image is clear and not blurry</Text>
                  <Text style={styles.errorSuggestionsText}>• Check that the image file is not corrupted</Text>
                  <Text style={styles.errorSuggestionsText}>• Try taking a new photo with better lighting</Text>
                </View>
              )}
              <TouchableOpacity
                style={styles.retakeButton}
                onPress={clearResults}
              >
                <Text style={styles.retakeButtonText}>📷 Retake Photo</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    );
  };

  const loadingRotation = loadingAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });



  const renderLiveCameraModal = () => {
    return (
      <Modal
        visible={showLiveCamera}
        animationType="slide"
        presentationStyle="fullScreen"
      >
        <View style={styles.liveCameraContainer}>
          {cameraPermission?.granted ? (
            <ExpoCameraView
              ref={cameraRef}
              style={styles.liveCameraView}
              facing={cameraType}
            />
          ) : (
            <View style={styles.liveCameraView}>
              <View style={styles.cameraPlaceholder}>
                <Text style={styles.cameraPlaceholderText}>📷</Text>
                <Text style={styles.cameraPlaceholderTitle}>Camera Permission Required</Text>
                <Text style={styles.cameraPlaceholderSubtitle}>
                  Please grant camera permission to use live preview
                </Text>
                <TouchableOpacity
                  style={styles.permissionButton}
                  onPress={openLiveCamera}
                >
                  <Text style={styles.permissionButtonText}>Grant Permission</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* TMT Bar Overlay */}
          {isOverlayVisible && (
            <Image
              source={overlaySource}
              style={[styles.overlayImage, { opacity: overlayOpacity }]}
              onLoad={() => console.log('TMT overlay loaded successfully')}
              onError={(error) => console.log('TMT overlay error:', error)}
            />
          )}

          {/* Camera Controls */}
          <View style={styles.liveCameraControls}>
            {/* Top Controls */}
            <View style={styles.topControls}>
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setShowLiveCamera(false)}
              >
                <Text style={styles.closeButtonText}>✕ Close</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.flipButton}
                onPress={() => setCameraType(cameraType === 'back' ? 'front' : 'back')}
              >
                <Text style={styles.flipButtonText}>🔄 Flip</Text>
              </TouchableOpacity>
            </View>

            {/* Overlay Controls */}
            <View style={styles.overlayControlsLive}>
              <TouchableOpacity
                style={styles.overlayToggleButton}
                onPress={() => setIsOverlayVisible(!isOverlayVisible)}
              >
                <Text style={styles.overlayToggleText}>
                  {isOverlayVisible ? 'Hide Guide' : 'Show Guide'}
                </Text>
              </TouchableOpacity>
              
              <View style={styles.opacityControlLive}>
                <Text style={styles.opacityLabel}>Opacity</Text>
                <Slider
                  style={styles.opacitySlider}
                  minimumValue={0}
                  maximumValue={1}
                  step={0.05}
                  value={overlayOpacity}
                  minimumTrackTintColor="#00D4FF"
                  maximumTrackTintColor="#FFFFFF"
                  onValueChange={setOverlayOpacity}
                />
              </View>
            </View>

            {/* Capture Button */}
            <View style={styles.captureControls}>
              <TouchableOpacity
                style={styles.captureButton}
                onPress={captureFromLiveCamera}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>
            </View>

            {/* Instructions */}
            <View style={styles.instructionsOverlay}>
              <Text style={styles.instructionText}>
                Position the TMT bar to match the guide overlay
              </Text>
              <Text style={styles.instructionSubtext}>
                Adjust opacity slider to see both the bar and guide clearly
              </Text>
            </View>
          </View>
        </View>
      </Modal>
    );
  };

  return (
    <View style={styles.container}>
      {renderServerStatus()}
      
      <View style={styles.stepIndicator}>
        <View style={styles.stepContainer}>
          <View style={[
            styles.stepCircle,
            analysisResults ? styles.stepCircleCompleted : styles.stepCircleActive
          ]}>
            <Text style={[
              styles.stepNumber,
              analysisResults ? styles.stepNumberCompleted : styles.stepNumberActive
            ]}>1</Text>
          </View>
          <Text style={[
            styles.stepLabel,
            analysisResults ? styles.stepLabelCompleted : styles.stepLabelActive
          ]}>Complete Analysis</Text>
        </View>
      </View>
      
      <View style={styles.cameraPreview}>
        {renderCameraView()}
        
        <Text style={styles.stepInfo}>
          Unified Rib Analysis
        </Text>
      </View>
      
      <View style={styles.bottomSection}>
        {!capturedImage ? (
          <View style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={takePhoto}
            >
              <Text style={styles.cameraButtonText}>📷 Take Photo</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={pickImage}
            >
              <Text style={styles.cameraButtonText}>🖼️ Select Image</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.analysisControls}>
            {showCalibrationUI && quickDetectionResults ? (
              // Show calibrate button instead of analyze button when results are available
              <TouchableOpacity
                style={[styles.calibrateButton, isAnalyzing && styles.disabledButton]}
                onPress={startCalibration}
                disabled={isAnalyzing}
              >
                <Text style={styles.calibrateButtonText}>
                  {isAnalyzing ? '⏳ Calibrating...' : '🔬 Start SAM Calibration'}
                </Text>
              </TouchableOpacity>
            ) : showSegmentedImage && tmtDetectionResult ? (
              // Show analyze button for post-calibration analysis - ONLY if calibration is complete
              <TouchableOpacity
                style={[
                  styles.analyzeButton, 
                  (isAnalyzing || !tmtDetectionResult.used_scale_factor || !tmtDetectionResult.cropped_image) && styles.disabledButton
                ]}
                onPress={startAnalysisAfterCalibration}
                disabled={isAnalyzing || !tmtDetectionResult.used_scale_factor || !tmtDetectionResult.cropped_image}
              >
                <Text style={styles.analyzeButtonText}>
                  {isAnalyzing ? '⏳ Analyzing...' : 
                   !tmtDetectionResult.used_scale_factor ? '[ERROR] Calibration Incomplete' :
                   !tmtDetectionResult.cropped_image ? '[ERROR] Segmentation Failed' :
                   '🔬 Complete Rib Analysis'}
                </Text>
              </TouchableOpacity>
            ) : (
              // Show analyze button normally
              <TouchableOpacity
                style={[styles.analyzeButton, isAnalyzing && styles.disabledButton]}
                onPress={() => {
                  if (imageUri) {
                    analyzeImage(''); // This will now trigger the base64 conversion
                  }
                }}
                disabled={isAnalyzing}
              >
                <Text style={styles.analyzeButtonText}>
                  {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze'}
                </Text>
              </TouchableOpacity>
            )}
            
            <TouchableOpacity 
              style={styles.clearButton} 
              onPress={clearResults}
            >
              <Text style={styles.clearButtonText}>📷 Retake Photo</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
      
      {isAnalyzing && (
        <View style={styles.processingOverlay}>
          <View style={styles.processingContent}>
            <Text style={styles.processingIcon}>⏳</Text>
            <Text style={styles.processingText}>Processing...</Text>
            {detectionProgress ? (
              <Text style={styles.processingProgress}>{detectionProgress}</Text>
            ) : null}
          </View>
        </View>
      )}
      
      {/* YOLO Detection Overlay */}
      {isYoloDetecting && (
        <View style={styles.yoloDetectionOverlay}>
          <View style={styles.yoloDetectionContent}>
            <Text style={styles.yoloDetectionIcon}>🔍</Text>
            <Text style={styles.yoloDetectionText}>YOLO Detection</Text>
            <Text style={styles.yoloDetectionSubtext}>Detecting TMT bars and ribs...</Text>
            {detectionProgress ? (
              <Text style={styles.yoloDetectionProgress}>{detectionProgress}</Text>
            ) : null}
          </View>
        </View>
      )}
      
      {/* Calibration Results Popup */}
      {showCalibrationUI && quickDetectionResults && (
        <View style={styles.calibrationOverlay}>
          <View style={styles.calibrationPopup}>
            <View style={styles.calibrationHeader}>
              <Text style={styles.calibrationTitle}>
                [SUCCESS] Quick Detection Complete for Unified Analysis!
              </Text>
              <Text style={styles.calibrationSubtitle}>
                Ready for SAM Calibration for Complete Rib Analysis
              </Text>
            </View>
            
            <View style={styles.calibrationResults}>
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>TMT Bars Detected:</Text>
                <Text style={styles.resultValue}>{quickDetectionResults.tmtBars}</Text>
              </View>
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Total Ribs Found:</Text>
                <Text style={styles.resultValue}>{quickDetectionResults.totalRibs}</Text>
              </View>
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Ribs in TMT Region:</Text>
                <Text style={styles.resultValue}>{quickDetectionResults.ribCount}</Text>
              </View>
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>Detection Time:</Text>
                <Text style={styles.resultValue}>{quickDetectionResults.inferenceTime.toFixed(3)}s</Text>
              </View>
            </View>
            
            <View style={styles.calibrationStatus}>
              <Text style={styles.calibrationStatusText}>
                {quickDetectionResults.ribCount >= 10 ? 
                  '✅ Sufficient ribs detected for analysis' : 
                  '❌ Insufficient ribs for accurate analysis'
                }
              </Text>
            </View>
            
            {quickDetectionResults.ribCount >= 10 && (
              <TouchableOpacity
                style={styles.calibrateButton}
                onPress={startCalibration}
              >
                <Text style={styles.calibrateButtonText}>
                  🔬 Start SAM Calibration for Unified Analysis
                </Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      )}

      {/* Live Camera Modal */}
      {renderLiveCameraModal()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0A',
  },
  cameraPreview: {
    flex: 1,
    backgroundColor: '#1A1A2E',
    borderRadius: 12,
    overflow: 'hidden',
    margin: 8,
    minHeight: 350,
    maxHeight: 400,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  livePreviewContainer: {
    position: 'relative',
    width: '100%',
    height: 300,
    backgroundColor: '#000',
  },
  camera: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  overlayContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  overlayImage: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 400,
    height: 500,
    marginTop: -250, // Half of height to center vertically
    marginLeft: -200, // Half of width to center horizontally
    resizeMode: 'contain',
  },
  overlayControlsMini: {
    position: 'absolute',
    top: 8,
    right: 8,
    flexDirection: 'row',
    gap: 8,
  },
  smallButton: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  smallButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
  opacitySliderBar: {
    position: 'absolute',
    left: 8,
    right: 8,
    bottom: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8,
    padding: 8,
  },
  sliderLabel: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
  },
  sliderValue: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  slider: {
    width: '100%',
    height: 40,
  },
  sliderThumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: AppColors.primaryBlue,
  },
  sliderTrack: {
    height: 4,
    borderRadius: 2,
    backgroundColor: '#E0E0E0',
  },
  instructionTextSmall: {
    color: '#FFFFFF',
    fontSize: 12,
    textAlign: 'center',
    opacity: 0.8,
  },
  serverStatusContainer: {
    position: 'absolute',
    top: 16,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 8,
    borderRadius: 8,
  },
  serverStatusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 4,
    borderRadius: 4,
  },
  serverStatusIcon: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginRight: 4,
  },
  serverStatusText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  processingContent: {
    backgroundColor: '#16213E',
    padding: 30,
    borderRadius: 20,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#00D4FF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  processingIcon: {
    color: '#00D4FF',
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  processingText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  processingProgress: {
    color: '#00D4FF',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 8,
    opacity: 0.9,
  },
  yoloDetectionOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  yoloDetectionContent: {
    backgroundColor: '#1A1A2E',
    padding: 30,
    borderRadius: 20,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#FFD700',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  yoloDetectionIcon: {
    color: '#FFD700',
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  yoloDetectionText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  yoloDetectionSubtext: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 12,
  },
  yoloDetectionProgress: {
    color: '#FFD700',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    opacity: 0.9,
  },
  calibrationContainer: {
    backgroundColor: '#16213E',
    borderRadius: 16,
    padding: 20,
    margin: 16,
    borderWidth: 2,
    borderColor: '#00D4FF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  calibrationHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  calibrationTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  calibrationSubtitle: {
    color: '#00D4FF',
    fontSize: 16,
    fontWeight: '600',
  },
  calibrationResults: {
    marginBottom: 20,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  resultLabel: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },
  resultValue: {
    color: '#00D4FF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  calibrationStatus: {
    alignItems: 'center',
    marginBottom: 20,
  },
  calibrationStatusText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  calibrateButton: {
    backgroundColor: '#00D4FF',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
    elevation: 4,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  calibrateButtonText: {
    color: '#16213E',
    fontSize: 18,
    fontWeight: 'bold',
  },
  calibrationOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  calibrationPopup: {
    backgroundColor: '#16213E',
    borderRadius: 20,
    padding: 24,
    margin: 20,
    borderWidth: 2,
    borderColor: '#00D4FF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
    maxWidth: '90%',
    maxHeight: '80%',
  },
  rotationHandle: {
    position: 'absolute',
    top: 0,
    right: 0,
    padding: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 8,
  },
  rotationHandleText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resizeHandle: {
    position: 'absolute',
    width: 16,
    height: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 8,
  },
  resizeHandleTopLeft: {
    top: 0,
    left: 0,
  },
  resizeHandleTopRight: {
    top: 0,
    right: 0,
  },
  resizeHandleBottomLeft: {
    bottom: 0,
    left: 0,
  },
  resizeHandleBottomRight: {
    bottom: 0,
    right: 0,
  },
  adjustButton: {
    backgroundColor: AppColors.primaryBlue,
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    flex: 1,
    marginRight: 8,
    elevation: 2,
    shadowColor: AppColors.primaryBlue,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  adjustButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomSection: {
    backgroundColor: '#0A0A0A',
    paddingHorizontal: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 212, 255, 0.2)',
  },
  stepIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    backgroundColor: '#1A1A2E',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 212, 255, 0.2)',
  },
  stepContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  stepCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#E0E0E0',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#16213E',
  },
  stepCircleActive: {
    borderColor: AppColors.primaryBlue,
    backgroundColor: AppColors.primaryBlue,
  },
  stepCircleCompleted: {
    borderColor: AppColors.successGreen,
    backgroundColor: AppColors.successGreen,
  },
  stepCircleInactive: {
    borderColor: 'rgba(255, 255, 255, 0.3)',
    backgroundColor: '#16213E',
  },
  stepNumber: {
    fontSize: 14,
    fontWeight: 'bold',
    color: 'rgba(255, 255, 255, 0.7)',
  },
  stepNumberActive: {
    color: '#FFFFFF',
  },
  stepNumberCompleted: {
    color: '#FFFFFF',
  },
  stepNumberInactive: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  stepLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.7)',
    marginLeft: 8,
  },
  stepLabelActive: {
    color: '#00D4FF',
  },
  stepLabelCompleted: {
    color: AppColors.successGreen,
  },
  stepLabelInactive: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  stepConnector: {
    width: 40,
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    marginHorizontal: 16,
  },
  overlayBorder: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderWidth: 2,
    borderColor: '#FFD700',
    borderRadius: 4,
    borderStyle: 'dashed',
  },
  cameraControls: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 8,
    paddingVertical: 16,
  },
  cameraButton: {
    backgroundColor: AppColors.primaryBlue,
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    flex: 1,
    elevation: 2,
    shadowColor: AppColors.primaryBlue,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  cameraButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  activeButton: {
    backgroundColor: AppColors.successGreen,
  },
  inactiveButton: {
    backgroundColor: AppColors.textLight,
  },
  analysisInfo: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  analysisInfoText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  segmentedImageContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1A1A2E',
    borderRadius: 12,
    margin: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  segmentedImageHeader: {
    alignItems: 'center',
    marginBottom: 16,
  },
  segmentedImageTitle: {
    color: '#00D4FF',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  segmentedImageSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 16,
  },
  segmentedImage: {
    width: '100%',
    height: '60%',
    resizeMode: 'contain',
    borderRadius: 8,
    marginBottom: 16,
  },
  segmentedImageInfo: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  segmentedImageInfoText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  segmentedImageInstructions: {
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
    marginTop: 8,
  },
  segmentedImageInstructionsText: {
    color: '#00D4FF',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    lineHeight: 20,
  },
  detectionErrorOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 12,
  },
  detectionErrorContent: {
    backgroundColor: '#1A1A2E',
    padding: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FF6B6B',
    alignItems: 'center',
    maxWidth: '90%',
  },
  detectionErrorIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  detectionErrorTitle: {
    color: '#FF6B6B',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  detectionErrorText: {
    color: '#FFFFFF',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 20,
  },
  retakeButton: {
    backgroundColor: '#FF6B6B',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    elevation: 2,
    shadowColor: '#FF6B6B',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  retakeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  errorSuggestions: {
    marginTop: 16,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  errorSuggestionsTitle: {
    color: '#00D4FF',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  errorSuggestionsText: {
    color: '#FFFFFF',
    fontSize: 12,
    marginBottom: 4,
  },
  // Missing styles
  cameraPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
    backgroundColor: '#16213E',
  },
  cameraPlaceholderText: {
    fontSize: 64,
    marginBottom: 16,
  },
  cameraPlaceholderTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  cameraPlaceholderSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    marginBottom: 16,
    textAlign: 'center',
  },
  cameraPlaceholderInfo: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  cameraPlaceholderHint: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.6)',
    textAlign: 'center',
    opacity: 0.8,
  },
  imagePreviewContainer: {
    position: 'relative',
    width: '100%',
    height: 300,
    backgroundColor: '#000',
  },
  capturedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  stepInfo: {
    color: '#FFFFFF',
    fontSize: 16,
    marginTop: 8,
    opacity: 0.8,
    position: 'absolute',
    bottom: 16,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  analysisControls: {
    flexDirection: 'row',
    gap: 12,
    padding: 16,
  },
  disabledButton: {
    backgroundColor: AppColors.textLight,
  },
  analyzeButton: {
    backgroundColor: AppColors.successGreen,
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    flex: 1,
    elevation: 2,
    shadowColor: AppColors.successGreen,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  analyzeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  clearButton: {
    backgroundColor: AppColors.errorRed,
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    flex: 1,
    elevation: 2,
    shadowColor: AppColors.errorRed,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  clearButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  permissionButton: {
    backgroundColor: AppColors.primaryBlue,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 16,
  },
  permissionButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  // Live Camera Modal Styles
  liveCameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  liveCameraView: {
    flex: 1,
  },
  liveCameraControls: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 10,
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 50,
  },
  closeButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
  },
  closeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  flipButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
  },
  flipButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  overlayControlsLive: {
    position: 'absolute',
    top: 120,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 15,
    padding: 15,
  },
  overlayToggleButton: {
    backgroundColor: 'rgba(0, 212, 255, 0.8)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  overlayToggleText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: 'bold',
  },
  opacityControlLive: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  opacityLabel: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: 'bold',
    marginRight: 10,
    minWidth: 60,
  },
  opacitySlider: {
    flex: 1,
    height: 30,
  },
  captureControls: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderWidth: 4,
    borderColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFFFFF',
  },
  instructionsOverlay: {
    position: 'absolute',
    bottom: 150,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 10,
    padding: 15,
  },
  instructionText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 5,
  },
  instructionSubtext: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    textAlign: 'center',
  },
});

export default CameraView; 