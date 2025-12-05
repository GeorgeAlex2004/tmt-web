import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  Image,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';
import Svg, { Circle, Path, Text as SvgText } from 'react-native-svg';
import { BACKEND_URL } from '../config';

const { width } = Dimensions.get('window');

const ResultsScreen = ({ route, navigation }) => {
  const { imageUri, testId, diameter, analysisData: passedAnalysisData } = route.params; // Get analysisData from params

  const [loading, setLoading] = useState(!passedAnalysisData); // Only load if no data passed
  const [analysisData, setAnalysisData] = useState(passedAnalysisData || null);
  const [error, setError] = useState(null);

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;

  // Poll backend for results after upload (only if not passed directly)
  useEffect(() => {
    // If we already have analysis data, just animate
    if (passedAnalysisData) {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 0,
          duration: 600,
          useNativeDriver: true,
        }),
      ]).start();
      return;
    }

    let isMounted = true;

    const fetchResults = async () => {
      try {
        setLoading(true);
        setError(null);
        let data = null;
        let attempts = 0;
        const maxAttempts = 60; // 2 minutes max (60 x 2s)
        while (attempts < maxAttempts) {
          const res = await fetch(`${BACKEND_URL}/get-report?test_id=${testId}`);
          if (!res.ok) throw new Error('Network error');
          data = await res.json();
          if (data && !data.processing) break; // Wait until processing is done
          await new Promise(res => setTimeout(res, 2000));
          attempts++;
        }
        if (attempts === maxAttempts) throw new Error('Processing timed out.');
        if (isMounted) {
          setAnalysisData(data);
          setLoading(false);
          Animated.parallel([
            Animated.timing(fadeAnim, {
              toValue: 1,
              duration: 800,
              useNativeDriver: true,
            }),
            Animated.timing(slideAnim, {
              toValue: 0,
              duration: 600,
              useNativeDriver: true,
            }),
          ]).start();
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || 'Failed to fetch results');
          setLoading(false);
        }
      }
    };

    fetchResults();
    return () => { isMounted = false; };
  }, [testId, passedAnalysisData]);

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#0D47A1" />
        <Text style={{ marginTop: 16, fontSize: 16 }}>
          Processing image, please wait...
        </Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={{ color: 'red', fontSize: 16, marginBottom: 16 }}>
          {error}
        </Text>
        <Text
          style={styles.retryText}
          onPress={() => {
            setLoading(true);
            setError(null);
          }}
        >
          Tap to retry
        </Text>
      </View>
    );
  }

  if (!analysisData) return null;
  const { verdict, level1, level2, segmented_image_url, debug_image_url, segmented_image_base64, debug_image_base64 } = analysisData;

  // Display Level 1 results
  const level1Display = level1 ? (
    <View style={{ marginVertical: 10 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Level 1: Color & Shape Analysis</Text>
      <Text>Dark grey and light core visible: {level1.dark_grey_and_light_core_visible ? 'Yes' : 'No'}</Text>
      <Text>Continuous outer ring: {level1.continuous_outer_ring ? 'Yes' : 'No'}</Text>
      <Text>Concentric regions: {level1.concentric_regions ? 'Yes' : 'No'}</Text>
      <Text>Uniform thickness: {level1.uniform_thickness ? 'Yes' : 'No'}</Text>
    </View>
  ) : null;

  // Display Level 2 results
  const level2Display = level2 ? (
    <View style={{ marginVertical: 10 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Level 2: Dimensional Analysis</Text>
      <Text>Min Thickness (mm): {level2.min_thickness_mm}</Text>
      <Text>Max Thickness (mm): {level2.max_thickness_mm}</Text>
      <Text>Quality Status: {level2.quality_status ? 'PASS' : 'FAIL'}</Text>
      <Text>Message: {level2.quality_message}</Text>
    </View>
  ) : null;

  // Display images - use base64 if available, otherwise use URL
  const segmentedImage = (segmented_image_base64 || segmented_image_url) ? (
    <Image
      source={{ uri: segmented_image_base64 || `${BACKEND_URL}${segmented_image_url}` }}
      style={{ width: 300, height: 300, alignSelf: 'center', marginVertical: 10 }}
      resizeMode="contain"
    />
  ) : null;
  const debugImage = (debug_image_base64 || debug_image_url) ? (
    <Image
      source={{ uri: debug_image_base64 || `${BACKEND_URL}${debug_image_url}` }}
      style={{ width: 300, height: 300, alignSelf: 'center', marginVertical: 10 }}
      resizeMode="contain"
    />
  ) : null;

  // Helper for status and value display
  const ResultItem = ({ label, result, icon }) => (
    <Animated.View
      style={[
        styles.resultItem,
        {
          opacity: fadeAnim,
          transform: [{ translateX: slideAnim }],
        },
      ]}
    >
      <View style={styles.resultLeft}>
        <Icon name={icon} size={20} color="#666" style={styles.resultIcon} />
        <Text style={styles.resultLabel}>{label}</Text>
      </View>
      <View style={styles.resultRight}>
        <Text style={[
          styles.resultValue,
          result.status === 'pass'
            ? styles.pass
            : result.status === 'fail'
            ? styles.fail
            : styles.info
        ]}>
          {result.value}
        </Text>
        {result.status !== 'info' && (
          <Icon
            name={result.status === 'pass' ? 'check-circle' : 'cancel'}
            size={20}
            color={result.status === 'pass' ? '#4CAF50' : '#f44336'}
          />
        )}
      </View>
    </Animated.View>
  );

  // Compose level1/level2 results for display
  const level1Results = {
    layersDetected: {
      status: level1?.dark_grey_and_light_core_visible ? 'pass' : 'fail',
      value: level1?.dark_grey_and_light_core_visible ? 'Yes' : 'No',
    },
    continuousRing: {
      status: level1?.continuous_outer_ring ? 'pass' : 'fail',
      value: level1?.continuous_outer_ring ? 'Yes' : 'No',
    },
    concentricRegions: {
      status: level1?.concentric_regions ? 'pass' : 'fail',
      value: level1?.concentric_regions ? 'Yes' : 'No',
    },
    uniformThickness: {
      status: level1?.uniform_thickness ? 'pass' : 'fail',
      value: level1?.uniform_thickness ? 'Yes' : 'No',
    },
  };
  const level2Results = {
    rimThickness: {
      status: 'info',
      value: `${level2?.min_thickness_mm || 0} - ${level2?.max_thickness_mm || 0} mm`,
    },
    thicknessPercentage: {
      status: level2?.quality_status ? 'pass' : 'fail',
      value: `${((level2?.min_thickness_mm || 0) / diameter * 100).toFixed(1)} - ${((level2?.max_thickness_mm || 0) / diameter * 100).toFixed(1)}%`,
    },
    withinRange: {
      status: level2?.quality_status ? 'pass' : 'fail',
      value: level2?.quality_status ? 'Yes' : 'No',
    },
    meetsStandards: {
      status: level2?.quality_status ? 'pass' : 'fail',
      value: level2?.quality_status ? 'Yes' : 'No',
    },
  };

  // Visualization (unchanged)
  const LayerVisualization = () => {
    const centerX = 100;
    const centerY = 100;
    const rimRadius = 90;
    const transitionRadius = 65;
    const coreRadius = 40;

    return (
      <Svg width={200} height={200} viewBox="0 0 200 200">
        <Circle
          cx={centerX}
          cy={centerY}
          r={rimRadius}
          fill="#2C3E50"
          opacity={0.9}
        />
        <Circle
          cx={centerX}
          cy={centerY}
          r={transitionRadius}
          fill="#7F8C8D"
          opacity={0.9}
        />
        <Circle
          cx={centerX}
          cy={centerY}
          r={coreRadius}
          fill="#BDC3C7"
          opacity={0.9}
        />
        <SvgText x={centerX} y={30} fill="white" fontSize="12" textAnchor="middle">
          Rim (Dark)
        </SvgText>
        <SvgText x={centerX} y={centerY} fill="#2C3E50" fontSize="12" textAnchor="middle">
          Core
        </SvgText>
        <Path
          d={`M ${centerX} ${centerY} L ${centerX + rimRadius} ${centerY}`}
          stroke="#4CAF50"
          strokeWidth="2"
          strokeDasharray="5,5"
        />
        <SvgText
          x={centerX + 45}
          y={centerY - 5}
          fill="#4CAF50"
          fontSize="10"
          textAnchor="middle"
        >
          {`${level2?.min_thickness_mm || 0}-${level2?.max_thickness_mm || 0}mm`}
        </SvgText>
      </Svg>
    );
  };

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.gradientContainer}
    >
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={{ fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginVertical: 10, color: '#fff' }}>Ring Test Results</Text>
      {segmentedImage}
      {debugImage}
      <Animated.View
        style={[
          styles.statusCard,
          {
            opacity: fadeAnim,
            transform: [{ scale: fadeAnim }],
          },
        ]}
      >
        <LinearGradient
          colors={verdict === 'PASS' ? ['#4CAF50', '#388E3C'] : ['#f44336', '#d32f2f']}
          style={styles.statusGradient}
        >
          <Icon
            name={verdict === 'PASS' ? 'check-circle' : 'cancel'}
            size={60}
            color="white"
          />
          <Text style={styles.statusText}>Test Result: {verdict}</Text>
          <Text style={styles.diameterInfo}>Bar Diameter: {diameter} mm</Text>
        </LinearGradient>
      </Animated.View>

      {/* Level 1 Processed Image */}
      <View style={styles.imageSection}>
          <Text style={styles.sectionTitle}>Segmented Image</Text>
        {(segmented_image_base64 || segmented_image_url) ? (
          <Image
            source={{ uri: segmented_image_base64 || `${BACKEND_URL}${segmented_image_url}` }}
            style={styles.processedImage}
            resizeMode="contain"
          />
        ) : (
          <Text style={styles.value}>No segmented image available.</Text>
        )}
      </View>

      {/* Visualization */}
      <Animated.View
        style={[
          styles.visualizationCard,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        <Text style={styles.sectionTitle}>Layer Detection</Text>
        <View style={styles.visualizationContainer}>
          <LayerVisualization />
        </View>
        <Text style={styles.visualizationNote}>
          Cross-section analysis after NITOL application
        </Text>
      </Animated.View>

      <View style={styles.resultsSection}>
        <View style={styles.sectionHeader}>
          <Icon name="color-lens" size={24} color="#0D47A1" />
          <Text style={styles.sectionTitle}>Level 1: Color & Shape Analysis</Text>
        </View>
        <ResultItem label="Layers Detected" result={level1Results.layersDetected} icon="layers" />
        <ResultItem label="Outer Ring" result={level1Results.continuousRing} icon="panorama-fish-eye" />
        <ResultItem label="Concentricity" result={level1Results.concentricRegions} icon="adjust" />
        <ResultItem label="Thickness Uniformity" result={level1Results.uniformThickness} icon="straighten" />
      </View>

      {/* Level 2 Processed Image */}
      <View style={styles.imageSection}>
          <Text style={styles.sectionTitle}>Thickness Image</Text>
        {(debug_image_base64 || debug_image_url) ? (
          <Image
            source={{ uri: debug_image_base64 || `${BACKEND_URL}${debug_image_url}` }}
            style={styles.processedImage}
            resizeMode="contain"
          />
        ) : (
          <Text style={styles.value}>No thickness image available.</Text>
        )}
      </View>

      <View style={styles.resultsSection}>
        <View style={styles.sectionHeader}>
          <Icon name="straighten" size={24} color="#0D47A1" />
          <Text style={styles.sectionTitle}>Level 2: Dimensional Analysis</Text>
        </View>
        <ResultItem label="Rim Thickness" result={level2Results.rimThickness} icon="format-size" />
        <ResultItem label="Thickness Percentage" result={level2Results.thicknessPercentage} icon="percent" />
        <ResultItem label="Range Compliance" result={level2Results.withinRange} icon="rule" />
        <ResultItem label="Meets Standards" result={level2Results.meetsStandards} icon="verified" />
      </View>

      <View style={styles.actionContainer}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() =>
            navigation.navigate('Report', {
              diameter,
              analysisData,
            })
          }
        >
          <LinearGradient
            colors={['#0D47A1', '#1976D2']}
            style={styles.buttonGradient}
          >
            <Icon name="description" size={24} color="white" />
            <Text style={styles.buttonText}>Generate Report</Text>
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => navigation.navigate('DiameterSelection')}
        >
          <Icon name="add-circle-outline" size={24} color="#0D47A1" />
          <Text style={styles.secondaryButtonText}>New Test</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F5F5F5' },
  statusCard: {
    margin: 20,
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  statusGradient: { padding: 30, alignItems: 'center' },
  statusText: { fontSize: 24, fontWeight: 'bold', color: 'white', marginTop: 10 },
  diameterInfo: { fontSize: 16, color: 'rgba(255,255,255,0.9)', marginTop: 5 },
  visualizationCard: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    margin: 20,
    marginTop: 0,
    padding: 20,
    borderRadius: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
  },
  visualizationContainer: { alignItems: 'center', marginVertical: 20 },
  visualizationNote: { textAlign: 'center', color: '#666', fontSize: 12, fontStyle: 'italic' },
  resultsSection: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    margin: 20,
    marginTop: 0,
    padding: 20,
    borderRadius: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
  },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 20 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', color: '#fff', marginLeft: 10 },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.12)',
  },
  resultLeft: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  resultIcon: { marginRight: 10 },
  resultLabel: { fontSize: 16, color: '#fff' },
  resultRight: { flexDirection: 'row', alignItems: 'center' },
  resultValue: { fontSize: 16, fontWeight: '600', marginRight: 8 },
  pass: { color: '#4CAF50' },
  fail: { color: '#f44336' },
  info: { color: '#2196F3' },
  actionContainer: { padding: 20, paddingTop: 0 },
  primaryButton: {
    marginBottom: 10,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 30,
  },
  buttonText: { color: 'white', fontSize: 16, fontWeight: 'bold', marginLeft: 10 },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingVertical: 18,
    paddingHorizontal: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#0D47A1',
  },
  secondaryButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 10 },
  imageSection: { marginHorizontal: 20, marginTop: 10, marginBottom: 10 },
  processedImage: { width: '100%', height: 220, borderRadius: 8, backgroundColor: 'rgba(255,255,255,0.08)', marginBottom: 10 },
  value: { fontSize: 16, color: '#444', flexShrink: 1 },
  retryText: { color: '#1976D2', fontSize: 16, textDecorationLine: 'underline', marginTop: 10 },
  gradientContainer: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 40,
    paddingTop: 10,
    minHeight: '100%',
  },
});

export default ResultsScreen;
