import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Animated,
} from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const TestDetailsScreen = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const { testData } = route.params;

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;

  useEffect(() => {
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
  }, []);

  const handleGoBack = () => {
    navigation.goBack();
  };

  const isPassed = testData.status === 'Passed';

  // Mock detailed data if not provided
  const getDetailedData = () => {
    if (testData.testType === 'Ring Test') {
      return {
        level1: {
          darkGreyAndLightCoreVisible: true,
          continuousOuterRing: true,
          concentricRegions: true,
          uniformThickness: true,
        },
        level2: {
          minThicknessMm: 1.8,
          maxThicknessMm: 2.2,
          qualityStatus: isPassed,
          qualityMessage: isPassed ? 'Thickness within acceptable range' : 'Thickness outside acceptable range',
        },
      };
    } else {
      // Rib Test data
      return {
        angle: {
          value: 45.2,
          confidenceInterval: 2.1,
          stdDev: 1.8,
          measurements: 5,
          rawValues: [44.1, 45.3, 46.2, 44.8, 45.6],
        },
        length: {
          value: 12.5,
          mean: 12.3,
          confidenceInterval: 0.8,
          stdDev: 0.6,
          measurements: 4,
          rawValues: [12.1, 12.8, 12.3, 12.7],
        },
        height: {
          value: 2.1,
          mean: 2.15,
          confidenceInterval: 0.3,
          stdDev: 0.2,
          measurements: 3,
          rawValues: [2.0, 2.2, 2.1],
        },
        interdistance: {
          value: 8.5,
          mean: 8.4,
          confidenceInterval: 0.5,
          stdDev: 0.4,
          measurements: 4,
          rawValues: [8.2, 8.6, 8.3, 8.7],
        },
        arValue: 0.045,
      };
    }
  };

  const detailedData = getDetailedData();

  const renderRingTestDetails = () => (
    <View style={styles.detailsContainer}>
      <Text style={styles.sectionTitle}>Level 1 Analysis</Text>
      <View style={styles.analysisGrid}>
        <View style={styles.analysisItem}>
          <Icon name="check-circle" size={20} color="#4CAF50" />
          <Text style={styles.analysisText}>Dark grey and light core visible</Text>
        </View>
        <View style={styles.analysisItem}>
          <Icon name="check-circle" size={20} color="#4CAF50" />
          <Text style={styles.analysisText}>Continuous outer ring</Text>
        </View>
        <View style={styles.analysisItem}>
          <Icon name="check-circle" size={20} color="#4CAF50" />
          <Text style={styles.analysisText}>Concentric regions</Text>
        </View>
        <View style={styles.analysisItem}>
          <Icon name="check-circle" size={20} color="#4CAF50" />
          <Text style={styles.analysisText}>Uniform thickness</Text>
        </View>
      </View>
      <Text style={styles.sectionTitle}>Level 2 Analysis</Text>
      <View style={styles.measurementGrid}>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Min Thickness</Text>
          <Text style={styles.measurementValue}>{detailedData.level2.minThicknessMm} mm</Text>
        </View>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Max Thickness</Text>
          <Text style={styles.measurementValue}>{detailedData.level2.maxThicknessMm} mm</Text>
        </View>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Quality Status</Text>
          <Text style={[styles.measurementValue, { color: isPassed ? '#4CAF50' : '#F44336' }]}>
            {detailedData.level2.qualityStatus ? 'PASS' : 'FAIL'}
          </Text>
        </View>
      </View>
      <View style={styles.messageContainer}>
        <Text style={styles.messageText}>{detailedData.level2.qualityMessage}</Text>
      </View>
    </View>
  );

  const renderRibTestDetails = () => (
    <View style={styles.detailsContainer}>
      <Text style={styles.sectionTitle}>Rib Measurements</Text>
      <View style={styles.measurementGrid}>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Rib Angle</Text>
          <Text style={styles.measurementValue}>{detailedData.angle.value}°</Text>
          <Text style={styles.measurementSubtext}>±{detailedData.angle.confidenceInterval}°</Text>
        </View>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Rib Length</Text>
          <Text style={styles.measurementValue}>{detailedData.length.value} mm</Text>
          <Text style={styles.measurementSubtext}>±{detailedData.length.confidenceInterval} mm</Text>
        </View>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Rib Height</Text>
          <Text style={styles.measurementValue}>{detailedData.height.value} mm</Text>
          <Text style={styles.measurementSubtext}>±{detailedData.height.confidenceInterval} mm</Text>
        </View>
        <View style={styles.measurementItem}>
          <Text style={styles.measurementLabel}>Interdistance</Text>
          <Text style={styles.measurementValue}>{detailedData.interdistance.value} mm</Text>
          <Text style={styles.measurementSubtext}>±{detailedData.interdistance.confidenceInterval} mm</Text>
        </View>
      </View>
      <View style={styles.arContainer}>
        <Text style={styles.sectionTitle}>AR Value</Text>
        <View style={styles.arValueContainer}>
          <Text style={styles.arValue}>{detailedData.arValue}</Text>
          <Text style={styles.arLabel}>Relative Rib Area</Text>
        </View>
      </View>
      <Text style={styles.sectionTitle}>Statistical Data</Text>
      <View style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Standard Deviation (Angle)</Text>
          <Text style={styles.statValue}>{detailedData.angle.stdDev}°</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Measurements Taken</Text>
          <Text style={styles.statValue}>{detailedData.angle.measurements}</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Raw Values</Text>
          <Text style={styles.statValue}>{detailedData.angle.rawValues.join(', ')}°</Text>
        </View>
      </View>
    </View>
  );

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      <ScrollView
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          style={[
            styles.headerContent,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleGoBack}
          >
            <Icon name="arrow-back" size={24} color="#00D4FF" />
          </TouchableOpacity>
          <View style={styles.logoContainer}>
            <Icon
              name={testData.testType === 'Ring Test' ? 'science' : 'straighten'}
              size={50}
              color="#00D4FF"
            />
          </View>
          <Text style={styles.headerTitle}>{testData.testType}</Text>
          <Text style={styles.headerSubtitle}>
            {testData.diameter} • {testData.date} at {testData.time}
          </Text>
          <View
            style={[
              styles.statusBadge,
              { backgroundColor: isPassed ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)' },
            ]}
          >
            <Icon
              name={isPassed ? 'check-circle' : 'error'}
              size={20}
              color={isPassed ? '#4CAF50' : '#F44336'}
            />
            <Text
              style={[
                styles.statusText,
                { color: isPassed ? '#4CAF50' : '#F44336' },
              ]}
            >
              {testData.status}
            </Text>
          </View>
        </Animated.View>
        <Animated.View
          style={[
            styles.detailsSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          {testData.testType === 'Ring Test' ? renderRingTestDetails() : renderRibTestDetails()}
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 50,
  },
  headerContent: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  backButton: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 1,
  },
  logoContainer: {
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.7)',
    marginBottom: 20,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  detailsSection: {
    paddingHorizontal: 20,
  },
  detailsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 16,
    marginTop: 8,
  },
  analysisGrid: {
    marginBottom: 20,
  },
  analysisItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  analysisText: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    marginLeft: 12,
  },
  measurementGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  measurementItem: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  measurementLabel: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    marginBottom: 8,
    textAlign: 'center',
  },
  measurementValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  measurementSubtext: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.5)',
  },
  messageContainer: {
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
    marginTop: 10,
  },
  messageText: {
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
  },
  arContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  arValueContainer: {
    alignItems: 'center',
    backgroundColor: 'rgba(0, 212, 255, 0.08)',
    borderRadius: 10,
    padding: 10,
    marginTop: 8,
    marginBottom: 8,
  },
  arValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#00D4FF',
  },
  arLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
  },
  statsContainer: {
    marginTop: 10,
  },
  statItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  statLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.7)',
  },
  statValue: {
    fontSize: 14,
    color: '#FFFFFF',
  },
});

export default TestDetailsScreen; 