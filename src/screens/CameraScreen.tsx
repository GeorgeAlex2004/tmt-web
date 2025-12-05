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
  Modal,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import CameraView from '../components/CameraView';
import { ApiService } from '../services/ApiService';
import { AnalysisStep } from '../models/AnalysisStep';

const { width, height } = Dimensions.get('window');

const CameraScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const [selectedDiameter, setSelectedDiameter] = useState<number | null>(null);
  const [selectedRows, setSelectedRows] = useState<number>(2);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [showResults, setShowResults] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  const resultCardAnim = useRef(new Animated.Value(0)).current;

  const diameters = [8, 10, 12, 16, 20, 25, 32];

  useEffect(() => {
    initializeAnimations();
  }, []);

  useEffect(() => {
    if (showResults) {
      animateResults();
    }
  }, [showResults]);

  useEffect(() => {
    if (showResults && analysisResults) {
      // Results are ready to display
    }
  }, [showResults, analysisResults]);

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

  const animateResults = () => {
    Animated.timing(resultCardAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();
  };

  const handleDiameterSelect = (diameter: number) => {
    setSelectedDiameter(diameter);
  };

  const handleAnalysisComplete = async (results: any) => {
    // Handle clear results case (when user retakes photo)
    if (results && results.clear_results) {
      setIsAnalyzing(false);
      setAnalysisResults(null);
      setShowResults(false);
      return;
    }
    
    console.log('=== UNIFIED ANALYSIS COMPLETE ===');
    console.log('Results received:', results);
    
    setIsAnalyzing(false);
    setAnalysisResults(results);
    setShowResults(true);
  };

  const handleViewResults = () => {
    // Navigate to AR Value screen with unified results
    navigation.navigate('ARValue', {
      results: analysisResults,
      diameter: selectedDiameter,
      rows: selectedRows,
    });
  };

  const handleBack = () => {
    navigation.goBack();
  };

  const resetAnalysis = async () => {
    // Clear frontend API service cache
    ApiService.clearCache();
    
    // Try to clear backend cache as well
    await ApiService.clearBackendCache();
    
    // Reset all state variables
    setSelectedDiameter(null);
    setSelectedRows(2);
    setAnalysisResults(null);
    setShowResults(false);
    resultCardAnim.setValue(0);
    
    // Reset CameraView state by passing a key prop that changes on reset
    setResetKey(prev => prev + 1);
    
    // Show feedback that cache was cleared
    Alert.alert(
      'Reset Complete',
      'All analysis data and cache have been cleared. You can start fresh with a new analysis.',
      [{ text: 'OK' }]
    );
  };

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const resultCardScale = resultCardAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.8, 1],
  });

  const resultCardOpacity = resultCardAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

    return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      {/* Header */}
      <Animated.View 
        style={[
          styles.header,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          }
        ]}
      >
        <View style={styles.headerContent}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleBack}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>TMT Analysis</Text>
          <TouchableOpacity
            style={styles.resetButton}
            onPress={resetAnalysis}
          >
            <Text style={styles.resetButtonText}>🔄 Reset</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>

      <ScrollView 
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Analysis Info */}
        <Animated.View 
          style={[
            styles.stepInfoSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <View style={styles.stepInfoCard}>
            <Text style={styles.stepInfoIcon}>🔬</Text>
            <Text style={styles.stepInfoTitle}>Complete Rib Analysis</Text>
            <Text style={styles.stepInfoDesc}>Analyze angle, height, length, interdistance, and calculate AR value from a single image</Text>
          </View>
        </Animated.View>

        {/* Configuration Section */}
        {!showResults && (
          <Animated.View 
            style={[
              styles.configSection,
              {
                opacity: fadeAnim,
                transform: [{ translateY: slideAnim }],
              }
            ]}
          >
            {/* Diameter Selection */}
            {!selectedDiameter && (
              <View style={styles.configCard}>
                <Text style={styles.configTitle}>📏 Select TMT Bar Diameter</Text>
                <Text style={styles.configSubtitle}>Choose the diameter of your TMT bar for accurate analysis</Text>
                <View style={styles.diameterGrid}>
                  {diameters.map((diameter) => (
                    <TouchableOpacity
                      key={diameter}
                      style={styles.diameterButton}
                      onPress={() => handleDiameterSelect(diameter)}
                      activeOpacity={0.8}
                    >
                      <Text style={styles.diameterText}>{diameter}mm</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            {/* Selected Configuration Display */}
            {selectedDiameter && (
              <View style={styles.selectedConfigCard}>
                <View style={styles.selectedConfigRow}>
                  <Text style={styles.selectedConfigIcon}>📏</Text>
                  <Text style={styles.selectedConfigText}>{selectedDiameter}mm Diameter</Text>
                </View>
                <View style={styles.selectedConfigRow}>
                  <Text style={styles.selectedConfigIcon}>📊</Text>
                  <Text style={styles.selectedConfigText}>2 Rows (Standard)</Text>
                </View>
                <TouchableOpacity
                  style={styles.changeConfigButton}
                  onPress={() => {
                    setSelectedDiameter(null);
                  }}
                >
                  <Text style={styles.changeConfigButtonText}>✏️ Change</Text>
                </TouchableOpacity>
              </View>
            )}
          </Animated.View>
        )}

        {/* Camera View */}
        {selectedDiameter && !showResults && (
          <Animated.View 
            style={[
              styles.cameraSection,
              {
                opacity: fadeAnim,
                transform: [{ translateY: slideAnim }],
              }
            ]}
          >
            <View style={styles.cameraCard}>
              <Text style={styles.cameraTitle}>📷 Capture Image</Text>
              <Text style={styles.cameraSubtitle}>
                Position your TMT bar and capture an image for complete rib analysis
              </Text>
              <CameraView
                key={resetKey}
                onAnalysisComplete={handleAnalysisComplete}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
                diameter={selectedDiameter}
                rows={selectedRows}
              />
            </View>
          </Animated.View>
        )}

        {/* Results Section */}
        {showResults && analysisResults && (
          <Animated.View 
            style={[
              styles.resultsSection,
              {
                opacity: resultCardOpacity,
                transform: [{ scale: resultCardScale }],
              }
            ]}
          >
            <View style={styles.resultsCard}>
              <View style={styles.resultsHeader}>
                <Text style={styles.resultsIcon}>✅</Text>
                <Text style={styles.resultsTitle}>Complete Analysis Results!</Text>
              </View>
              
              <View style={styles.resultsContent}>
                {/* Primary Results Grid */}
                <View style={styles.resultsSection}>
                  <Text style={styles.sectionTitle}>📊 Measured Parameters</Text>
                  
                  <View style={styles.resultGrid}>
                    <View style={styles.resultCard}>
                      <Text style={styles.resultIcon}>📐</Text>
                      <Text style={styles.resultLabel}>Rib Angle</Text>
                      <Text style={styles.resultValue}>
                        {analysisResults.angle?.value ? analysisResults.angle.value.toFixed(2) : 'N/A'}°
                      </Text>
                      <Text style={styles.resultSubtext}>
                        ±{analysisResults.angle?.confidence_interval ? analysisResults.angle.confidence_interval.toFixed(2) : 'N/A'}°
                      </Text>
                    </View>
                    
                    <View style={styles.resultCard}>
                      <Text style={styles.resultIcon}>📏</Text>
                      <Text style={styles.resultLabel}>Rib Height</Text>
                      <Text style={styles.resultValue}>
                        {analysisResults.height?.value ? analysisResults.height.value.toFixed(2) : 'N/A'}mm
                      </Text>
                      <Text style={styles.resultSubtext}>
                        ±{analysisResults.height?.confidence_interval ? analysisResults.height.confidence_interval.toFixed(2) : 'N/A'}mm
                      </Text>
                    </View>
                  </View>
                  
                  <View style={styles.resultGrid}>
                    <View style={styles.resultCard}>
                      <Text style={styles.resultIcon}>📐</Text>
                      <Text style={styles.resultLabel}>Rib Length</Text>
                      <Text style={styles.resultValue}>
                        {analysisResults.length?.value ? analysisResults.length.value.toFixed(2) : 'N/A'}mm
                      </Text>
                      <Text style={styles.resultSubtext}>
                        ±{analysisResults.length?.confidence_interval ? analysisResults.length.confidence_interval.toFixed(2) : 'N/A'}mm
                      </Text>
                    </View>
                    
                    <View style={styles.resultCard}>
                      <Text style={styles.resultIcon}>📏</Text>
                      <Text style={styles.resultLabel}>Interdistance</Text>
                      <Text style={styles.resultValue}>
                        {analysisResults.interdistance?.value ? analysisResults.interdistance.value.toFixed(2) : 'N/A'}mm
                      </Text>
                      <Text style={styles.resultSubtext}>
                        ±{analysisResults.interdistance?.confidence_interval ? analysisResults.interdistance.confidence_interval.toFixed(2) : 'N/A'}mm
                      </Text>
                    </View>
                  </View>
                </View>

                {/* AR Value Display */}
                {analysisResults.ar_value && (
                  <View style={styles.resultsSection}>
                    <Text style={styles.sectionTitle}>🎯 AR Value</Text>
                    <View style={styles.arValueDisplay}>
                      <Text style={styles.arValueNumber}>{analysisResults.ar_value.toFixed(4)}</Text>
                      <Text style={styles.arValueLabel}>Relative Rib Area</Text>
                    </View>
                  </View>
                )}

                {/* Analysis Details */}
                <View style={styles.resultsSection}>
                  <Text style={styles.sectionTitle}>🔍 Analysis Details</Text>
                  
                  <View style={styles.detailsGrid}>
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Processing Time:</Text>
                      <Text style={styles.detailValue}>
                        {analysisResults.processing_time ? `${analysisResults.processing_time}s` : 'N/A'}
                      </Text>
                    </View>
                    
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Scale Factor:</Text>
                      <Text style={styles.detailValue}>
                        {analysisResults.used_scale_factor ? analysisResults.used_scale_factor.toFixed(4) : 'N/A'}
                      </Text>
                    </View>
                    
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>TMT Diameter:</Text>
                      <Text style={styles.detailValue}>
                        {analysisResults.diameter || 'N/A'}mm
                      </Text>
                    </View>
                    
                    <View style={styles.detailItem}>
                      <Text style={styles.detailLabel}>Calibration Method:</Text>
                      <Text style={styles.detailValue}>
                        {analysisResults.calibration_method || 'N/A'}
                      </Text>
                    </View>
                  </View>
                </View>
              </View>
              
              <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
                <TouchableOpacity
                  style={styles.nextStepButton}
                  onPress={handleViewResults}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={['#4ECDC4', '#44A08D']}
                    style={styles.nextStepGradient}
                  >
                    <Text style={styles.nextStepIcon}>📊</Text>
                    <Text style={styles.nextStepText}>View Detailed Report</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </Animated.View>
            </View>
          </Animated.View>
        )}
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
  scrollContent: {
    paddingBottom: 40,
  },
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 10,
  },
  backButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  resetButton: {
    padding: 10,
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  progressSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  progressStep: {
    alignItems: 'center',
    flex: 1,
  },
  progressCircle: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  progressIcon: {
    fontSize: 24,
  },
  progressTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  progressLine: {
    position: 'absolute',
    top: 30,
    left: '50%',
    width: '100%',
    height: 2,
    zIndex: -1,
  },
  stepInfoSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  stepInfoCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  stepInfoIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  debugText: {
    color: '#FFD700',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 10,
    fontFamily: 'monospace',
  },
  stepInfoTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  stepInfoDesc: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    lineHeight: 22,
  },
  configSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  configCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  configTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  configSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 20,
  },
  diameterGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  diameterButton: {
    width: '30%',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 15,
    padding: 15,
    marginBottom: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  diameterText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  selectedConfigCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  selectedConfigRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  selectedConfigIcon: {
    fontSize: 20,
    marginRight: 12,
  },
  selectedConfigText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  changeConfigButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 15,
    alignSelf: 'center',
    marginTop: 12,
  },
  changeConfigButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  cameraSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  cameraCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  cameraTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  cameraSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 20,
  },
  resultsSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 16,
  },
  resultGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  resultCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 12,
    padding: 16,
    flex: 1,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  resultIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  resultLabel: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: '500',
    textAlign: 'center',
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 20,
    color: '#FFFFFF',
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 4,
  },
  resultSubtext: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  detailsGrid: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  detailItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  detailLabel: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: '500',
    flex: 1,
  },
  detailValue: {
    fontSize: 14,
    color: '#FFFFFF',
    fontWeight: 'bold',
    textAlign: 'right',
    flex: 1,
  },
  resultsCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  resultsIcon: {
    fontSize: 32,
    marginRight: 12,
  },
  resultsTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  resultsContent: {
    marginBottom: 20,
  },
  nextStepButton: {
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  nextStepGradient: {
    paddingVertical: 16,
    paddingHorizontal: 30,
    alignItems: 'center',
  },
  nextStepIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  nextStepText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  arValueDisplay: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  arValueNumber: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  arValueLabel: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
  },
});

export default CameraScreen; 