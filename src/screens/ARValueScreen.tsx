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
  Share,
} from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { AnalysisCalculator } from '../models/AnalysisCalculator';
import { PdfService } from '../services/PdfService';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width, height } = Dimensions.get('window');

// Convert height to mm using either px/mm or mm/px scale; pick a plausible result
const getHeightMmFromResults = (angleResults: any, heightResults: any, diameter: number): number => {
  const rawValue = heightResults?.height?.value ?? 0;
  const scale = heightResults?.used_scale_factor ?? angleResults?.used_scale_factor ?? 0;
  if (!rawValue || !scale) return rawValue || 0;
  const mm_if_px_per_mm = rawValue / scale; // scale = px/mm
  const mm_if_mm_per_px = rawValue * scale; // scale = mm/px
  const maxPlausible = Math.max(1, (diameter || 0) * 0.2) || 2; // cap relative to diameter, fallback 2mm
  const minPlausible = 0.05;
  const candidates = [mm_if_px_per_mm, mm_if_mm_per_px].filter(v => v > 0);
  const plausible = candidates.filter(v => v >= minPlausible && v <= maxPlausible);
  if (plausible.length === 1) return plausible[0];
  if (plausible.length > 1) return Math.min(...plausible);
  return Math.min(...candidates);
};

// Thresholds for rib height by bar diameter (mm)
const getHeightThresholdForDiameter = (diameter: number): { min: number; max: number } | null => {
  const thresholds: Record<number, { min: number; max: number }> = {
    12: { min: 0.5, max: 0.6 },
  };
  return thresholds[diameter] || null;
};

const ARValueDetailedScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const [arValue, setArValue] = useState<number | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [complianceStatus, setComplianceStatus] = useState<string>('');
  const [complianceColor, setComplianceColor] = useState<string>('');
  const [mockHeight, setMockHeight] = useState<number | null>(null);
  const [mockInterdistance, setMockInterdistance] = useState<number | null>(null);
  
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const resultAnim = useRef(new Animated.Value(0)).current;
  const pdfAnim = useRef(new Animated.Value(0)).current;

  const { results, diameter, rows } = route.params || {};
  
  // Extract individual results from unified analysis
  const angleResults = results || {};
  const heightResults = results || {};

  useEffect(() => {
    initializeAnimations();
    if (results && results.angle && results.height) {
      calculateARValue();
    }
    // No need for mock values since we get real results from unified analysis
    setMockHeight(null);
    setMockInterdistance(null);
  }, [results]);

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

  const calculateARValue = async () => {
    setIsCalculating(true);
    try {
      // Check if AR value is already calculated in the unified results
      if (results?.ar_value) {
        setArValue(results.ar_value);
      } else {
        // Fallback: Calculate AR value using the calculator if not provided
        const ribAngle = results?.angle?.value || 0;
        const ribHeight = results?.height?.value || 0;
        
        const arValue = AnalysisCalculator.calculateARValue({
          ribAngle,
          ribHeight,
          barDiameter: diameter,
        });
        
        setArValue(arValue);
      }
      
      const finalArValue = results?.ar_value || arValue;
      
      // Determine compliance status
      let complianceStatus = 'Non-Compliant';
      let complianceColor = '#ff6b6b';
      
      if (finalArValue >= 0.065) {
        complianceStatus = 'Compliant';
        complianceColor = '#51cf66';
      } else if (finalArValue >= 0.060) {
        complianceStatus = 'Marginally Compliant';
        complianceColor = '#ffd43b';
      }
      
      setComplianceStatus(complianceStatus);
      setComplianceColor(complianceColor);
      
      // Animate results
      Animated.timing(resultAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }).start();
    } catch (error) {
      Alert.alert('Error', 'Failed to calculate AR value');
    } finally {
      setIsCalculating(false);
    }
  };

  const generatePDF = async () => {
    setIsGeneratingPdf(true);
    try {
      // Extract values from the unified results
      const ribAngle = results?.angle?.value || 0;
      const ribHeight = results?.height?.value || 0;
      const ribLength = results?.length?.value || 0;
      const interdistance = results?.interdistance?.value || 0;
      
      // Format data for PDF service
      const analysisResults = {
        angle: ribAngle,
        height: ribHeight,
        length: ribLength,
        interdistance: interdistance,
        ar_value: results?.ar_value || arValue || 0,
        image_path: results?.image_path,
        scale_factor: results?.used_scale_factor,
        calibration_method: results?.calibration_method,
        processing_time: results?.processing_time,
      };
      
      // Generate PDF
      await PdfService.generateReport({
        analysisResults,
        diameter,
        numRows: rows,
      });
      
      // Animate PDF success
      Animated.timing(pdfAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }).start();
      
      Alert.alert(
        'PDF Generated!',
        'Report has been generated and shared successfully.',
        [{ text: 'OK', style: 'default' }]
      );
    } catch (error) {
      console.error('PDF generation error:', error);
      Alert.alert('Error', 'Failed to generate PDF report. Please try again.');
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const sharePDF = async (pdfPath: string) => {
    try {
      await Share.share({
        url: pdfPath,
        title: 'TMT Analysis Report',
        message: 'Here is your TMT bar analysis report',
      });
    } catch (error) {
      Alert.alert('Error', 'Failed to share PDF');
    }
  };

  const handleBack = () => {
    navigation.goBack();
  };

  const handleNewAnalysis = () => {
    navigation.navigate('Home');
  };

  const resultScale = resultAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.8, 1],
  });

  const resultOpacity = resultAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  const pdfScale = pdfAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.9, 1],
  });

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      {/* Header */}
      <Animated.View 
        style={[
            styles.headerContent,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          }
        ]}
      >
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleBack}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AR Value Results</Text>
          <TouchableOpacity
            style={styles.newAnalysisButton}
            onPress={handleNewAnalysis}
          >
            <Text style={styles.newAnalysisButtonText}>🔄 New</Text>
          </TouchableOpacity>
      </Animated.View>

      <ScrollView 
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Success Banner */}
        <Animated.View 
          style={[
            styles.successBanner,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <View style={styles.successContent}>
            <Text style={styles.successIcon}>🎉</Text>
            <Text style={styles.successTitle}>Analysis Complete!</Text>
            <Text style={styles.successSubtitle}>
              Your TMT bar has been analyzed successfully
            </Text>
          </View>
        </Animated.View>

        {/* AR Value Result */}
        {arValue !== null && (
          <Animated.View 
            style={[
              styles.arValueSection,
              {
                opacity: resultOpacity,
                transform: [{ scale: resultScale }],
              }
            ]}
          >
            <View style={styles.arValueCard}>
              <Text style={styles.arValueIcon}>🎯</Text>
              <Text style={styles.arValueTitle}>AR Value</Text>
              <Text style={styles.arValueNumber}>{arValue.toFixed(2)}</Text>
              <View style={[
                styles.complianceBadge,
                { backgroundColor: complianceColor }
              ]}>
                <Text style={styles.complianceText}>{complianceStatus}</Text>
              </View>
            </View>
          </Animated.View>
        )}

        {/* Analysis Summary */}
        <Animated.View 
          style={[
            styles.summarySection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <Text style={styles.sectionTitle}>Analysis Summary</Text>
          
          <View style={styles.summaryGrid}>
            <View style={styles.summaryCard}>
              <Text style={styles.summaryIcon}>📐</Text>
              <Text style={styles.summaryTitle}>Angle Analysis</Text>
                              <Text style={styles.summaryValue}>
                  {results?.angle?.value?.toFixed(2) || 'N/A'}°
                </Text>
              <Text style={styles.summaryLabel}>Average Rib Angle</Text>
            </View>
            
            <View style={styles.summaryCard}>
              <Text style={styles.summaryIcon}>📏</Text>
              <Text style={styles.summaryTitle}>Height Analysis</Text>
              <Text style={styles.summaryValue}>
                {results?.height?.value?.toFixed(2) || 'N/A'}mm
              </Text>
              <Text style={styles.summaryLabel}>Average Rib Height</Text>
            </View>
            
            <View style={styles.summaryCard}>
              <Text style={styles.summaryIcon}>📊</Text>
              <Text style={styles.summaryTitle}>Configuration</Text>
              <Text style={styles.summaryValue}>{diameter}mm</Text>
              <Text style={styles.summaryLabel}>{rows} Row{rows > 1 ? 's' : ''}</Text>
            </View>
            
            <View style={styles.summaryCard}>
              <Text style={styles.summaryIcon}>⚡</Text>
              <Text style={styles.summaryTitle}>Confidence</Text>
              <Text style={styles.summaryValue}>
                {Math.min(
                  results?.angle?.confidence_interval ? 100 - results.angle.confidence_interval : 95,
                  results?.height?.confidence_interval ? 100 - results.height.confidence_interval : 95
                ).toFixed(1)}%
              </Text>
              <Text style={styles.summaryLabel}>Overall Accuracy</Text>
            </View>
          </View>
        </Animated.View>

        {/* Detailed Results */}
        <Animated.View 
          style={[
            styles.detailsSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <Text style={styles.sectionTitle}>Detailed Measurements</Text>
          
          <View style={styles.detailsContainer}>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Rib Angle:</Text>
              <Text style={styles.detailValue}>
                {results?.angle?.value?.toFixed(2) || 'N/A'}°
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Interdistance:</Text>
              <Text style={styles.detailValue}>
                {results?.interdistance?.value?.toFixed(2) || 'N/A'}mm
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Rib Height:</Text>
              <Text style={styles.detailValue}>
                {results?.height?.value?.toFixed(2) || 'N/A'}mm
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Rib Length:</Text>
              <Text style={styles.detailValue}>
                {results?.length?.value?.toFixed(2) || 'N/A'}mm
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Bar Diameter:</Text>
              <Text style={styles.detailValue}>{diameter}mm</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Number of Rows:</Text>
              <Text style={styles.detailValue}>{rows}</Text>
          </View>
        </View>
        </Animated.View>

        {/* Compliance section removed as per request */}

        {/* Action Buttons */}
        <Animated.View 
          style={[
            styles.actionsSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
            <TouchableOpacity
              style={styles.generatePdfButton}
              onPress={generatePDF}
              disabled={isGeneratingPdf}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={['#4ECDC4', '#44A08D']}
                style={styles.pdfButtonGradient}
              >
                <Text style={styles.pdfButtonIcon}>
                  {isGeneratingPdf ? '⏳' : '📄'}
                </Text>
                <Text style={styles.pdfButtonText}>
                  {isGeneratingPdf ? 'Generating PDF...' : 'Generate PDF Report'}
                </Text>
              </LinearGradient>
          </TouchableOpacity>
          </Animated.View>
          
          <TouchableOpacity
            style={styles.shareButton}
            onPress={() => sharePDF('')}
            activeOpacity={0.8}
          >
            <Text style={styles.shareButtonIcon}>📤</Text>
            <Text style={styles.shareButtonText}>Share Results</Text>
          </TouchableOpacity>
        </Animated.View>

        {/* Trust Indicators */}
        <Animated.View 
          style={[
            styles.trustSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            }
          ]}
        >
          <View style={styles.trustCard}>
            <Text style={styles.trustTitle}>🔒 Your Data is Secure</Text>
            <Text style={styles.trustDesc}>
              All analysis data is processed locally and never stored on external servers. Your measurements remain private.
            </Text>
        </View>
        </Animated.View>
    </ScrollView>
    </LinearGradient>
  );
};

const ARValueSimpleScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const [arValue, setArValue] = useState<number | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [complianceStatus, setComplianceStatus] = useState<string>('');
  const [complianceColor, setComplianceColor] = useState<string>('');
  const [mockHeight, setMockHeight] = useState<number | null>(null);
  const [mockInterdistance, setMockInterdistance] = useState<number | null>(null);
  
  const { results, diameter, rows } = route.params || {};

  useEffect(() => {
    if (results && results.angle && results.height) {
      calculateARValue();
    }
    // No need for mock values since we get real results from unified analysis
    setMockHeight(null);
    setMockInterdistance(null);
  }, [results]);

  const calculateARValue = async () => {
    setIsCalculating(true);
    try {
      // Check if AR value is already calculated in the unified results
      let finalArValue;
      if (results?.ar_value) {
        finalArValue = results.ar_value;
        setArValue(finalArValue);
      } else {
        // Fallback: Calculate AR value using the calculator if not provided
        const ribAngle = results?.angle?.value || 0;
        const ribHeight = results?.height?.value || 0;
        
        finalArValue = AnalysisCalculator.calculateARValue({
          ribAngle,
          ribHeight,
          barDiameter: diameter,
        });
        
        setArValue(finalArValue);
      }
      
      // Determine compliance status
      let complianceStatus = 'Non-Compliant';
      let complianceColor = '#ff6b6b';
      
      if (finalArValue >= 0.065) {
        complianceStatus = 'Compliant';
        complianceColor = '#51cf66';
      } else if (finalArValue >= 0.060) {
        complianceStatus = 'Marginally Compliant';
        complianceColor = '#ffd43b';
      }
      
      setComplianceStatus(complianceStatus);
      setComplianceColor(complianceColor);
      
    } catch (error) {
      Alert.alert('Error', 'Failed to calculate AR value');
    } finally {
      setIsCalculating(false);
    }
  };

  const generatePDF = async () => {
    setIsGeneratingPdf(true);
    try {
      // Extract values from the unified results
      const ribAngle = results?.angle?.value || 0;
      const ribHeight = results?.height?.value || 0;
      const ribLength = results?.length?.value || 0;
      const interdistance = results?.interdistance?.value || 0;
      
      // Format data for PDF service
      const analysisResults = {
        angle: ribAngle,
        height: ribHeight,
        length: ribLength,
        interdistance: interdistance,
        ar_value: results?.ar_value || arValue || 0,
        image_path: results?.image_path,
        scale_factor: results?.used_scale_factor,
        calibration_method: results?.calibration_method,
        processing_time: results?.processing_time,
      };
      
      // Generate PDF
      await PdfService.generateReport({
        analysisResults,
        diameter,
        numRows: rows,
      });
      
      Alert.alert(
        'PDF Generated!',
        'Report has been generated and shared successfully.',
        [{ text: 'OK', style: 'default' }]
      );
    } catch (error) {
      console.error('PDF generation error:', error);
      Alert.alert('Error', 'Failed to generate PDF report. Please try again.');
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const sharePDF = async (pdfPath: string) => {
    try {
      await Share.share({
        url: pdfPath,
        title: 'TMT Analysis Report',
        message: 'Here is your TMT bar analysis report',
      });
    } catch (error) {
      Alert.alert('Error', 'Failed to share PDF');
    }
  };

  const handleBack = () => {
    navigation.goBack();
  };

  const handleNewAnalysis = () => {
    navigation.navigate('Home');
  };

  const handleMoreInfo = () => {
    navigation.navigate('ARValueDetailed', {
      results,
      diameter,
      rows,
    });
  };

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      {/* Header */}
      <View 
        style={styles.headerContent}
      >
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleBack}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AR Value</Text>
          <TouchableOpacity
            style={styles.newAnalysisButton}
            onPress={handleNewAnalysis}
          >
            <Text style={styles.newAnalysisButtonText}>🔄 New</Text>
          </TouchableOpacity>
      </View>

      <View style={styles.flexContainer}>
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
        {/* Success Banner */}
        <View style={styles.successBanner}>
          <View style={styles.successContent}>
            <Text style={styles.successIcon}>🎉</Text>
            <Text style={styles.successTitle}>Analysis Complete!</Text>
            <Text style={styles.successSubtitle}>
              Your TMT bar has been analyzed successfully
            </Text>
          </View>
        </View>
        {/* AR Value Result */}
        {arValue !== null && (
          <View 
            style={styles.arValueSection}
          >
            <View style={styles.arValueCard}>
              <Text style={styles.arValueIcon}>🎯</Text>
              <Text style={styles.arValueTitle}>AR Value</Text>
              <Text style={styles.arValueNumber}>{arValue.toFixed(2)}</Text>
              <View style={[
                styles.complianceBadge,
                { backgroundColor: complianceColor }
              ]}>
                <Text style={styles.complianceText}>{complianceStatus}</Text>
              </View>
            </View>
          </View>
        )}

        {/* Action Buttons */}
        <View 
          style={styles.actionsSection}
        >
          <TouchableOpacity
            style={styles.generatePdfButton}
            onPress={generatePDF}
            disabled={isGeneratingPdf}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={['#4ECDC4', '#44A08D']}
              style={styles.pdfButtonGradient}
            >
              <View style={styles.pdfButtonIcon}>
                {isGeneratingPdf ? (
                  <Icon name="hourglass-empty" size={24} color="#fff" />
                ) : (
                  <Icon name="picture-as-pdf" size={24} color="#fff" />
                )}
              </View>
              <Text style={styles.pdfButtonText}>
                {isGeneratingPdf ? 'Generating PDF...' : 'Generate PDF Report'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.shareButton}
            onPress={() => sharePDF('')}
            activeOpacity={0.8}
          >
            <Text style={styles.shareButtonIcon}>📤</Text>
            <Text style={styles.shareButtonText}>Share Results</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.moreInfoButton, { marginTop: 16 }]}
            onPress={handleMoreInfo}
            activeOpacity={0.8}
          >
            <Text style={styles.moreInfoButtonText}>More Information</Text>
          </TouchableOpacity>
        </View>
        {/* Trust Indicators at the bottom */}
        <View style={styles.trustSectionSticky}>
          <View style={styles.trustCard}>
            <Text style={styles.trustTitle}>🔒 Your Data is Secure</Text>
            <Text style={styles.trustDesc}>
              All analysis data is processed locally and never stored on external servers. Your measurements remain private.
            </Text>
          </View>
        </View>
        </ScrollView>
      </View>
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
  newAnalysisButton: {
    padding: 10,
  },
  newAnalysisButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  successBanner: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  successContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  successIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  successTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  successSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    lineHeight: 22,
  },
  arValueSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  arValueCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 30,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  arValueIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  arValueTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 12,
  },
  arValueNumber: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 16,
  },
  complianceBadge: {
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  complianceText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  summarySection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 20,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  summaryCard: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  summaryIcon: {
    fontSize: 32,
    marginBottom: 12,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  summaryValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 4,
  },
  summaryLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
  },
  detailsSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  detailsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  detailLabel: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    fontWeight: '500',
  },
  detailValue: {
    fontSize: 16,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  complianceSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  complianceCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  complianceIcon: {
    fontSize: 32,
    marginBottom: 12,
    textAlign: 'center',
  },
  complianceTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 12,
  },
  complianceDesc: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 20,
  },
  requirementRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  requirementLabel: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  requirementValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  actionsSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
    alignItems: 'center',
  },
  generatePdfButton: {
    borderRadius: 25,
    overflow: 'hidden',
    marginBottom: 16,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  pdfButtonGradient: {
    paddingVertical: 16,
    paddingHorizontal: 30,
    alignItems: 'center',
  },
  pdfButtonIcon: {
    marginBottom: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pdfButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  shareButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 20,
    paddingVertical: 12,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
  },
  shareButtonIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  shareButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  moreInfoButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 20,
    paddingVertical: 12,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  moreInfoButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  trustSection: {
    paddingHorizontal: 20,
  },
  trustCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  trustTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  trustDesc: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    lineHeight: 20,
  },
  flexContainer: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  trustSectionSticky: {
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: 'transparent',
  },
});

export { ARValueSimpleScreen, ARValueDetailedScreen }; 